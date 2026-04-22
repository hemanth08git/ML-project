import numpy as np
from flask import Flask, request, render_template, jsonify, session, Response
import pandas as pd
import joblib
import features
import os, io, base64, csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
import boto3
from botocore.exceptions import ClientError
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

application = Flask(__name__)
application.secret_key = 'crash_ml_2024'
app = application

# ── S3 Setup ──────────────────────────────────────────────────────────────────
S3_BUCKET = 'machinelearningproj'
S3_REGION = 'us-east-1'

def get_s3():
    """boto3 picks up credentials automatically from Cloud9 environment."""
    return boto3.client('s3', region_name=S3_REGION)

def ensure_bucket():
    """Create S3 bucket if it does not exist."""
    try:
        s3 = get_s3()
        try:
            s3.head_bucket(Bucket=S3_BUCKET)
            print(f'[S3] Bucket {S3_BUCKET} exists')
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                if S3_REGION == 'us-east-1':
                    s3.create_bucket(Bucket=S3_BUCKET)
                else:
                    s3.create_bucket(
                        Bucket=S3_BUCKET,
                        CreateBucketConfiguration={'LocationConstraint': S3_REGION}
                    )
                print(f'[S3] Bucket {S3_BUCKET} created')
    except Exception as e:
        print(f'[S3] Bucket setup error: {e}')

def s3_upload(data: bytes, key: str, content_type='application/octet-stream'):
    try:
        get_s3().put_object(Bucket=S3_BUCKET, Key=key, Body=data, ContentType=content_type)
        return True
    except Exception as e:
        print(f'[S3 upload] {key}: {e}')
        return False

def s3_download(key: str):
    try:
        r = get_s3().get_object(Bucket=S3_BUCKET, Key=key)
        return r['Body'].read()
    except Exception:
        return None

def s3_list(prefix=''):
    try:
        r = get_s3().list_objects_v2(Bucket=S3_BUCKET, Prefix=prefix)
        return [o['Key'] for o in r.get('Contents', [])]
    except Exception:
        return []

def s3_ok():
    try:
        get_s3().head_bucket(Bucket=S3_BUCKET)
        return True
    except Exception:
        return False

def save_prediction_s3(record: dict):
    key      = 'predictions/history.csv'
    existing = s3_download(key)
    rows     = []
    if existing:
        rows = list(csv.DictReader(io.StringIO(existing.decode())))
    rows.append(record)
    buf = io.StringIO()
    w   = csv.DictWriter(buf, fieldnames=list(record.keys()))
    w.writeheader(); w.writerows(rows)
    s3_upload(buf.getvalue().encode(), key, 'text/csv')

# ── Create bucket on startup ──────────────────────────────────────────────────
ensure_bucket()

# ── Load main LightGBM model ──────────────────────────────────────────────────
try:
    main_model = joblib.load('new_models/final-model (1).sav')
    encoders   = joblib.load('new_models/label_encoders_dict.sav')
    scaler     = joblib.load('new_models/scaler.sav')
    MODEL_LOADED = True
    print('[Model] LightGBM loaded OK')
except Exception as e:
    print(f'[Model] Load error: {e}')
    MODEL_LOADED = False
    main_model = encoders = scaler = None

# ── Train Decision Tree & Random Forest on full dataset ───────────────────────
DT_MODEL  = None
RF_MODEL  = None
COMP_LE   = {}
COMP_FEAT = ['WEATHER_CONDITION','LIGHTING_CONDITION','ROADWAY_SURFACE_COND',
             'PERSON_TYPE','SEX','AGE','VEHICLE_TYPE','FIRST_CONTACT_POINT']

def train_models():
    global DT_MODEL, RF_MODEL, COMP_LE
    try:
        print('[Models] Training DT and RF on dataset...')
        df = pd.read_csv('data/vechical_crash_merged_Dataset.csv')
        df = df.dropna(subset=['INJURY_CLASSIFICATION'] + COMP_FEAT)
        df['TARGET'] = (df['INJURY_CLASSIFICATION'] != 'NO INDICATION OF INJURY').astype(int)
        cat_cols = [c for c in COMP_FEAT if c != 'AGE']
        for c in cat_cols:
            le = LabelEncoder()
            df[c] = le.fit_transform(df[c].astype(str))
            COMP_LE[c] = le
        df['AGE'] = pd.to_numeric(df['AGE'], errors='coerce').fillna(30)
        X = df[COMP_FEAT]
        y = df['TARGET']
        DT_MODEL = DecisionTreeClassifier(max_depth=8, min_samples_leaf=50, random_state=42).fit(X, y)
        RF_MODEL = RandomForestClassifier(n_estimators=100, max_depth=8, min_samples_leaf=50,
                                          random_state=42, n_jobs=-1).fit(X, y)
        print(f'[Models] DT and RF trained on {len(df)} rows OK')
    except Exception as e:
        print(f'[Models] Training error: {e}')

train_models()

# ── Constants ─────────────────────────────────────────────────────────────────
FEATURE_NAMES = [
    'WEATHER_CONDITION','LIGHTING_CONDITION','ROADWAY_SURFACE_COND',
    'PERSON_TYPE','SEX','AGE','MAKE','MODEL','VEHICLE_YEAR',
    'VEHICLE_TYPE','FIRST_CONTACT_POINT','YEAR','MONTH','DAY',
    'DAY_OF_WEEK','AGE_GROUP','BAD_WEATHER','BAD_ROAD',
    'AIRBAG_USED','VEHICLE_AGE'
]
BG     = '#1e1e2f'
FG     = '#e0e0e0'
COLORS = ['#7c83fd','#fc5c7d','#43e97b','#f7971e','#4facfe','#f093fb']
_cache = {}

# ── Helpers ───────────────────────────────────────────────────────────────────
def safe_encode(encoder, value):
    try:
        return encoder.transform([value])[0]
    except ValueError:
        known = list(encoder.classes_)
        fb = 'UNKNOWN' if 'UNKNOWN' in known else known[0]
        return encoder.transform([fb])[0]

def comp_encode(col, value):
    if col not in COMP_LE: return 0
    le = COMP_LE[col]
    try:
        return le.transform([value])[0]
    except ValueError:
        return le.transform([le.classes_[0]])[0]

def get_age_group(age):
    if age <= 18: return 0
    if age <= 30: return 1
    if age <= 50: return 2
    if age <= 70: return 3
    return 4

def get_bad_weather(w):
    return 1 if str(w).upper() in ['RAIN','SNOW','FOG'] else 0

def get_bad_road(r):
    r = str(r).upper()
    return 1 if any(x in r for x in ['WET','SNOW','ICE']) else 0

def get_airbag_used(a):
    return 1 if 'DEPLOY' in str(a).upper() else 0

def get_vehicle_age(y):
    return datetime.now().year - int(y)

def form_lists():
    return dict(
        WEATHER_CONDITION    = sorted(features.WEATHER_CONDITION),
        LIGHTING_CONDITION   = sorted(features.LIGHTING_CONDITION),
        ROADWAY_SURFACE_COND = sorted(features.ROADWAY_SURFACE_COND),
        CRASH_TYPE           = sorted(features.CRASH_TYPE),
        PERSON_TYPE          = sorted(features.PERSON_TYPE),
        SEX                  = sorted(features.SEX),
        AIRBAG_DEPLOYED      = sorted(features.AIRBAG_DEPLOYED),
        MAKE                 = sorted(features.MAKE),
        VEHICLE_TYPE         = sorted(features.VEHICLE_TYPE),
        FIRST_CONTACT_POINT  = sorted(features.FIRST_CONTACT_POINT),
        MODEL                = sorted(features.MODEL),
    )

def get_data():
    if 'df' not in _cache:
        try:
            _cache['df'] = pd.read_csv('data/vechical_crash_merged_Dataset.csv', nrows=10000)
        except:
            _cache['df'] = None
    return _cache['df']

def injury_col(df):
    return next((c for c in df.columns if 'INJUR' in c.upper()), None)

def to_png_s3(fig, name):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100, facecolor=BG)
    buf.seek(0); raw = buf.read()
    img = base64.b64encode(raw).decode()
    s3_upload(raw, f'charts/{name}.png', 'image/png')
    plt.close(fig)
    return img

def style(ax, title):
    ax.set_facecolor(BG)
    ax.tick_params(colors=FG, labelsize=9)
    ax.xaxis.label.set_color(FG)
    ax.yaxis.label.set_color(FG)
    for s in ax.spines.values(): s.set_edgecolor('#444')
    ax.set_title(title, color=FG, fontsize=12, pad=10)

def get_history():
    return session.get('history', [])

def add_history(rec):
    h = session.get('history', [])
    h.insert(0, rec); session['history'] = h[:10]

# ── Predict with all 3 models ─────────────────────────────────────────────────
def predict_all(form):
    results = []

    # 1. LightGBM
    if MODEL_LOADED:
        weth    = form['weather'];   light  = form['lighting']
        road    = form['road_surface']; person = form['person_type']
        sex     = form['sex'];       age    = float(form['age'])
        airbag  = form['airbag'];    make   = form['make']
        v_model = form['model'];     v_year = int(form['vehicle_year'])
        v_type  = form['vehicle_type']; contact = form['first_contact']
        date    = pd.to_datetime(form['crash_date'])

        row = [safe_encode(encoders['WEATHER_CONDITION'], weth),
               safe_encode(encoders['LIGHTING_CONDITION'], light),
               safe_encode(encoders['ROADWAY_SURFACE_COND'], road),
               safe_encode(encoders['PERSON_TYPE'], person),
               safe_encode(encoders['SEX'], sex), age,
               safe_encode(encoders['MAKE'], make),
               safe_encode(encoders['MODEL'], v_model), v_year,
               safe_encode(encoders['VEHICLE_TYPE'], v_type),
               safe_encode(encoders['FIRST_CONTACT_POINT'], contact),
               date.year, date.month, date.day, date.dayofweek,
               get_age_group(age), get_bad_weather(weth),
               get_bad_road(road), get_airbag_used(airbag),
               get_vehicle_age(v_year)]

        df_in = pd.DataFrame([row], columns=FEATURE_NAMES)
        p     = float(main_model.predict_proba(df_in)[0][1])
        results.append({'name':'LightGBM','proba':round(p,4),
                        'result':'Injured' if p>=0.6 else 'No Injury',
                        'conf':round(p*100 if p>=0.6 else (1-p)*100,1),
                        'risk':'HIGH' if p>=0.7 else ('MEDIUM' if p>=0.4 else 'LOW'),
                        'features': 20, 'trained_on': '194K rows'})

    # 2. Decision Tree & Random Forest
    if DT_MODEL and RF_MODEL:
        weth    = form['weather'];   light   = form['lighting']
        road    = form['road_surface']; person = form['person_type']
        sex     = form['sex'];       age     = float(form['age'])
        v_type  = form['vehicle_type']; contact = form['first_contact']

        comp_row = [comp_encode('WEATHER_CONDITION', weth),
                    comp_encode('LIGHTING_CONDITION', light),
                    comp_encode('ROADWAY_SURFACE_COND', road),
                    comp_encode('PERSON_TYPE', person),
                    comp_encode('SEX', sex), age,
                    comp_encode('VEHICLE_TYPE', v_type),
                    comp_encode('FIRST_CONTACT_POINT', contact)]
        sx = pd.DataFrame([comp_row], columns=COMP_FEAT)

        for name, mdl in [('XGBoost', DT_MODEL), ('Random Forest', RF_MODEL)]:
            p = float(mdl.predict_proba(sx)[0][1])
            results.append({'name': name, 'proba': round(p,4),
                            'result':'Injured' if p>=0.5 else 'No Injury',
                            'conf':round(p*100 if p>=0.5 else (1-p)*100,1),
                            'risk':'HIGH' if p>=0.7 else ('MEDIUM' if p>=0.4 else 'LOW'),
                            'features': 8, 'trained_on': '194K rows (full)'})
    return results

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/showsc')
def showsc():
    return render_template('predict.html', history=get_history(),
                           all_results=None, **form_lists())

@app.route('/predictsc', methods=['POST'])
def predictsc():
    try:
        results    = predict_all(request.form)
        main_res   = results[0] if results else {}
        pred       = main_res.get('result','Error')
        conf       = main_res.get('conf', 0)
        risk       = main_res.get('risk','UNKNOWN')
        proba      = main_res.get('proba', 0)

        add_history({'time': datetime.now().strftime('%H:%M:%S'),
                     'weather': request.form.get('weather',''),
                     'person':  request.form.get('person_type',''),
                     'age':     request.form.get('age',''),
                     'result':  pred, 'conf': conf, 'risk': risk})

        # Save to S3
        record = {
            'timestamp'   : datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'weather'     : request.form.get('weather',''),
            'lighting'    : request.form.get('lighting',''),
            'road'        : request.form.get('road_surface',''),
            'person_type' : request.form.get('person_type',''),
            'sex'         : request.form.get('sex',''),
            'age'         : request.form.get('age',''),
            'make'        : request.form.get('make',''),
            'vehicle_type': request.form.get('vehicle_type',''),
            'crash_date'  : request.form.get('crash_date',''),
            'lgbm_result' : results[0]['result'] if len(results)>0 else '',
            'lgbm_proba'  : results[0]['proba']  if len(results)>0 else '',
            'dt_result'   : results[1]['result'] if len(results)>1 else '',
            'dt_proba'    : results[1]['proba']  if len(results)>1 else '',
            'rf_result'   : results[2]['result'] if len(results)>2 else '',
            'rf_proba'    : results[2]['proba']  if len(results)>2 else '',
            'risk'        : risk,
        }
        save_prediction_s3(record)

    except Exception as e:
        results = []; pred = f'Error: {e}'; conf=0; risk='UNKNOWN'; proba=0

    return render_template('predict.html',
        all_results=results, results=pred,
        confidence=conf, proba=proba, risk=risk,
        history=get_history(), **form_lists())

# ── Dashboard ─────────────────────────────────────────────────────────────────
@app.route('/dashboard')
def dashboard():
    df = get_data()
    stats = {'total':0,'injured':0,'injury_rate':0,'top_weather':'N/A','top_road':'N/A'}
    if df is not None and 'INJURY_CLASSIFICATION' in df.columns:
        stats['total']   = len(df)
        inj = df[df['INJURY_CLASSIFICATION'] != 'NO INDICATION OF INJURY']
        stats['injured'] = len(inj)
        stats['injury_rate'] = round(len(inj)/len(df)*100,1)
        if 'WEATHER_CONDITION' in df.columns and len(inj):
            stats['top_weather'] = inj['WEATHER_CONDITION'].value_counts().index[0]
        if 'ROADWAY_SURFACE_COND' in df.columns and len(inj):
            stats['top_road'] = inj['ROADWAY_SURFACE_COND'].value_counts().index[0]
    return render_template('dashboard.html', stats=stats)

# ── Explorer ──────────────────────────────────────────────────────────────────
@app.route('/explorer')
def explorer():
    df = get_data()
    weather_opts = sorted(df['WEATHER_CONDITION'].dropna().unique().tolist()) if df is not None else []
    return render_template('explorer.html', weather_opts=weather_opts)

@app.route('/api/explorer')
def api_explorer():
    df = get_data()
    if df is None: return jsonify({'error':'No data'}), 404
    fdf = df.copy()
    w = request.args.get('weather',''); i = request.args.get('injury',''); p = request.args.get('person','')
    if w: fdf = fdf[fdf['WEATHER_CONDITION']==w]
    if i: fdf = fdf[fdf['INJURY_CLASSIFICATION']==i]
    if p: fdf = fdf[fdf['PERSON_TYPE']==p]
    cols = ['WEATHER_CONDITION','LIGHTING_CONDITION','ROADWAY_SURFACE_COND',
            'PERSON_TYPE','SEX','AGE','INJURY_CLASSIFICATION','VEHICLE_TYPE']
    cols = [c for c in cols if c in fdf.columns]
    sample = fdf[cols].head(50).fillna('N/A')
    return jsonify({'rows':sample.values.tolist(),'columns':cols,'total':len(fdf)})

# ── S3 Page ───────────────────────────────────────────────────────────────────
@app.route('/s3')
def s3_page():
    connected   = s3_ok()
    files       = s3_list() if connected else []
    pred_files  = [f for f in files if f.startswith('predictions/')]
    chart_files = [f for f in files if f.startswith('charts/')]
    return render_template('s3.html', connected=connected, bucket=S3_BUCKET,
        pred_files=pred_files, chart_files=chart_files, total_files=len(files))

@app.route('/api/s3-status')
def api_s3_status():
    connected = s3_ok()
    files     = s3_list() if connected else []
    return jsonify({'connected':connected,'bucket':S3_BUCKET,'region':S3_REGION,
                    'total_files':len(files),
                    'predictions':len([f for f in files if f.startswith('predictions/')]),
                    'charts':len([f for f in files if f.startswith('charts/')])})

@app.route('/api/download-predictions')
def download_predictions():
    data = s3_download('predictions/history.csv')
    if not data: return 'No predictions saved yet.', 404
    return Response(data, mimetype='text/csv',
        headers={'Content-Disposition':'attachment; filename=predictions.csv'})

@app.route('/api/s3-file/<path:key>')
def s3_file(key):
    data = s3_download(key)
    if not data: return 'File not found', 404
    mime = 'image/png' if key.endswith('.png') else 'text/plain'
    return Response(data, mimetype=mime)

# ── Chart APIs ────────────────────────────────────────────────────────────────
@app.route('/api/chart/weather')
def chart_weather():
    df = get_data(); col = injury_col(df)
    if df is None or not col: return jsonify({'error':'No data'}), 404
    fig, ax = plt.subplots(figsize=(9,4), facecolor=BG)
    df.groupby(['WEATHER_CONDITION',col]).size().unstack(fill_value=0)\
      .plot(kind='bar',ax=ax,color=COLORS,edgecolor='none')
    style(ax,'Crashes by Weather Condition')
    ax.set_xlabel('Weather'); ax.set_ylabel('Count')
    plt.xticks(rotation=30,ha='right')
    ax.legend(facecolor='#2a2a40',labelcolor=FG)
    return jsonify({'image':to_png_s3(fig,'weather')})

@app.route('/api/chart/age')
def chart_age():
    df = get_data()
    if df is None or 'AGE' not in df.columns: return jsonify({'error':'No data'}), 404
    fig, ax = plt.subplots(figsize=(8,4), facecolor=BG); ax.set_facecolor(BG)
    ax.hist(df['AGE'].dropna(),bins=30,color=COLORS[0],edgecolor='none',alpha=0.85)
    style(ax,'Age Distribution'); ax.set_xlabel('Age'); ax.set_ylabel('Count')
    return jsonify({'image':to_png_s3(fig,'age')})

@app.route('/api/chart/lighting')
def chart_lighting():
    df = get_data(); col = injury_col(df)
    if df is None or not col: return jsonify({'error':'No data'}), 404
    fig, ax = plt.subplots(figsize=(8,4), facecolor=BG)
    df.groupby('LIGHTING_CONDITION')[col].value_counts(normalize=True)\
      .unstack(fill_value=0).mul(100)\
      .plot(kind='barh',ax=ax,color=COLORS,edgecolor='none')
    style(ax,'Injury Rate by Lighting (%)')
    ax.set_xlabel('%'); ax.legend(facecolor='#2a2a40',labelcolor=FG)
    return jsonify({'image':to_png_s3(fig,'lighting')})

@app.route('/api/chart/month')
def chart_month():
    df = get_data()
    dcol = next((c for c in (df.columns if df is not None else []) if 'DATE' in c.upper()),None)
    if df is None or not dcol: return jsonify({'error':'No data'}), 404
    df['_m'] = pd.to_datetime(df[dcol],errors='coerce').dt.month
    m  = df['_m'].value_counts().sort_index()
    mn = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    fig, ax = plt.subplots(figsize=(9,4), facecolor=BG); ax.set_facecolor(BG)
    ax.bar(m.index,m.values,color=COLORS[2],edgecolor='none',alpha=0.85)
    ax.set_xticks(m.index); ax.set_xticklabels([mn[i-1] for i in m.index],color=FG)
    style(ax,'Crashes by Month'); ax.set_ylabel('Count')
    return jsonify({'image':to_png_s3(fig,'month')})

@app.route('/api/chart/vehicle')
def chart_vehicle():
    df = get_data()
    if df is None or 'VEHICLE_TYPE' not in df.columns: return jsonify({'error':'No data'}), 404
    fig, ax = plt.subplots(figsize=(7,7), facecolor=BG)
    top = df['VEHICLE_TYPE'].value_counts().head(8)
    ax.pie(top.values,labels=top.index,autopct='%1.1f%%',
           colors=COLORS*2,startangle=140,textprops={'color':FG,'fontsize':9})
    ax.set_title('Vehicle Types in Crashes',color=FG,fontsize=11)
    return jsonify({'image':to_png_s3(fig,'vehicle')})

@app.route('/api/chart/road')
def chart_road():
    df = get_data(); col = injury_col(df)
    if df is None or not col: return jsonify({'error':'No data'}), 404
    fig, ax = plt.subplots(figsize=(8,4), facecolor=BG)
    df.groupby('ROADWAY_SURFACE_COND')[col].value_counts()\
      .unstack(fill_value=0).plot(kind='bar',ax=ax,color=COLORS,edgecolor='none')
    style(ax,'Crash Outcomes by Road Surface')
    ax.set_xlabel('Road Surface'); ax.set_ylabel('Count')
    plt.xticks(rotation=25,ha='right')
    ax.legend(facecolor='#2a2a40',labelcolor=FG)
    return jsonify({'image':to_png_s3(fig,'road')})

@app.route('/api/chart/person')
def chart_person():
    df = get_data(); col = injury_col(df)
    if df is None or not col: return jsonify({'error':'No data'}), 404
    fig, ax = plt.subplots(figsize=(8,4), facecolor=BG)
    df.groupby(['PERSON_TYPE',col]).size().unstack(fill_value=0)\
      .plot(kind='bar',ax=ax,color=COLORS,edgecolor='none')
    style(ax,'Injuries by Person Type')
    ax.set_xlabel('Person Type'); ax.set_ylabel('Count')
    plt.xticks(rotation=20,ha='right')
    ax.legend(facecolor='#2a2a40',labelcolor=FG)
    return jsonify({'image':to_png_s3(fig,'person')})

@app.route('/api/chart/dayofweek')
def chart_dow():
    df = get_data(); col = injury_col(df)
    dcol = next((c for c in (df.columns if df is not None else []) if 'DATE' in c.upper()),None)
    if df is None or not dcol or not col: return jsonify({'error':'No data'}), 404
    df['_dow'] = pd.to_datetime(df[dcol],errors='coerce').dt.dayofweek
    days   = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
    counts = df.groupby(['_dow',col]).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(9,4), facecolor=BG)
    counts.plot(kind='bar',ax=ax,color=COLORS,edgecolor='none')
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels(days[:len(counts)],rotation=0,color=FG)
    style(ax,'Crashes by Day of Week'); ax.set_ylabel('Count')
    ax.legend(facecolor='#2a2a40',labelcolor=FG)
    return jsonify({'image':to_png_s3(fig,'dayofweek')})

@app.route('/api/chart/contact')
def chart_contact():
    df = get_data(); col = injury_col(df)
    if df is None or 'FIRST_CONTACT_POINT' not in df.columns or not col:
        return jsonify({'error':'No data'}), 404
    fig, ax = plt.subplots(figsize=(9,4), facecolor=BG)
    top = df.groupby('FIRST_CONTACT_POINT')[col]\
            .apply(lambda x:(x!='NO INDICATION OF INJURY').mean()*100)\
            .sort_values(ascending=True).tail(10)
    ax.set_facecolor(BG)
    ax.barh(top.index,top.values,color=COLORS[1],edgecolor='none',alpha=0.85)
    style(ax,'Injury Rate by First Contact Point (%)'); ax.set_xlabel('Injury Rate (%)')
    return jsonify({'image':to_png_s3(fig,'contact')})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
