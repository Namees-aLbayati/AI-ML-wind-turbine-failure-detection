"""Build and execute the final submission using the supplied assignment structure."""
from pathlib import Path
import copy
import nbformat as nb
from nbclient import NotebookClient
from nbconvert import HTMLExporter
from model_selection_narrative import add_model_selection_interpretation
from submission_narrative import clean_submission
ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / 'notebooks/reference/AIML_Project_1_Full_Code_Notebook.ipynb'
DEST = ROOT / 'notebooks/AIML_Project_1_Full_Code_Notebook_Completed.ipynb'

def build():
    template = nb.read(TEMPLATE, 4)
    code = {}
    notes = {}
    code[10] = "# Dependencies are pinned in requirements-neural.txt.\n# Install before starting: python -m pip install -r requirements-neural.txt"
    notes[11] = 'This notebook uses the project environment. Run all cells in order. The original completed submission is preserved in `notebooks/archive/`. The supplied template determines the section names and required tasks; the grouped test split and selected engineered predictors are documented project adaptations. New results below supersede the archived submission for this revised analysis. Earlier test exposure means this is not a fresh independent external evaluation.'
    code[12] = '''from pathlib import Path
import os, sys, time, json
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, Markdown
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, fbeta_score, average_precision_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier
import tensorflow as tf
from tensorflow import keras
root = Path.cwd()
if not (root / 'data/raw').exists(): root = root.parent
sys.path.insert(0, str(root))
from src.modeling.artifacts import BalancedXGBClassifier
from src.modeling.neural_networks import dataset
sns.set_theme(style='whitegrid')
pd.set_option('display.max_columns', None)
RANDOM_STATE = 42
output = root / 'models/template_aligned'
output.mkdir(parents=True, exist_ok=True)
def observation(text): display(Markdown('**Observation:** ' + text))
'''
    code[15] = "data = pd.read_csv(root / 'data/raw/wind_turbine_detection.csv')\nprint('Loaded', len(data), 'SCADA records.')"
    for idx,src in {19:"display(data.head(5)); display(data.tail(5))",22:"print('Rows and columns:', data.shape)",25:"display(data.dtypes.rename('data_type').to_frame())",28:"display(data.describe().T)",31:"display(pd.DataFrame({'missing_count': data.isna().sum(), 'missing_pct': data.isna().mean()*100}))",34:"print('Duplicate rows:', data.duplicated().sum()); print('Duplicate turbine/timestamp keys:', data.duplicated(['turbine_id','timestamp']).sum())",37:"data['timestamp'] = pd.to_datetime(data['timestamp'])\ndata['dayofweek'] = data.timestamp.dt.dayofweek\ndata['hour'] = data.timestamp.dt.hour"}.items(): code[idx]=src
    notes[19]='Both ends of the source data are displayed to inspect its structure; these are records, not independent turbines or failure episodes.'
    notes[25]='Timestamp is initially read as text and converted below; turbine ID is categorical, failure is a binary target, and the remaining source fields are numeric.'
    notes[28]='Compare the quartiles and maxima to identify heavy tails. Extreme condition readings may represent faults and are retained rather than automatically removed.'
    notes[31]='Missingness is small and limited to three sensors. Partition-specific percentages and training-only imputation appear under Missing Value Treatment.'
    notes[34]='The source has no duplicate rows or duplicate turbine–timestamp keys; no duplicate removal is needed.'
    code[40]='''def histogram(data_df, col, **kwargs):
    fig, ax = plt.subplots(figsize=(9, 3.6))
    sns.histplot(data=data_df, x=col, bins=50, ax=ax)
    ax.set_title(kwargs.get('title', col)); plt.tight_layout(); plt.show()
    s = data_df[col].dropna()
    observation(f"{col}: middle 50% spans {s.quantile(.25):.3f}–{s.quantile(.75):.3f}; range {s.min():.3f}–{s.max():.3f}; skewness {s.skew():.3f}. Extreme values warrant domain review.")
def barchart(data_df, x_col, **kwargs):
    sns.countplot(data=data_df, x=x_col); plt.title(kwargs.get('title', x_col)); plt.show()
    display(data_df[x_col].value_counts().rename('records').to_frame())
def boxplot(data_df, x_col, y_col):
    sns.boxplot(data=data_df, x=x_col, y=y_col, showfliers=True); plt.show()
    summary = data_df.groupby(x_col)[y_col].agg(['median','mean','std','min','max'])
    display(summary)
    observation(f"{y_col}: normal median {summary.loc[0,'median']:.3f}, fault median {summary.loc[1,'median']:.3f}; standard deviations {summary.loc[0,'std']:.3f} and {summary.loc[1,'std']:.3f}. Overlap and outliers mean this is not a standalone fault threshold.")
def scatterplot(data_df, x_col, y_col):
    sns.scatterplot(data=data_df, x=x_col, y=y_col, s=4, alpha=.15)
    plt.show()
    observation(f"Pearson correlation is {data_df[[x_col,y_col]].corr().iloc[0,1]:.4f}. The scatter shows operating regimes and dispersion; correlation alone does not establish causality or identify invalid readings.")
'''
    features=['wind_speed_mps','air_density_kgm3','gearbox_oil_temp_C','generator_winding_temp_C','drivetrain_vibration_rms_mmps','tower_vibration_mmps','oil_particle_count','prior_fault_count','component_age_days']
    for idx,f in zip(range(46,71,3),features): code[idx]=f"histogram(data, {f!r})"
    notes[43]='The positive class is rare (about 3% of records). Accuracy alone would conceal missed failures, so recall and precision–recall performance guide model assessment.'
    code[73]="rates = data.groupby('turbine_id').failure.mean().mul(100)\nrates.plot.bar(ylabel='Failure rate (%)'); plt.show()\nobservation(f'Turbine failure rates range from {rates.min():.3f}% ({rates.idxmin()}) to {rates.max():.3f}% ({rates.idxmax()}).')"
    code[76]="wind_bins = pd.cut(data.wind_speed_mps, bins=np.arange(0, data.wind_speed_mps.max()+2, 2), include_lowest=True)\npower_curve = data.groupby(wind_bins, observed=True).power_output_kW.agg(['mean','count'])\ndisplay(power_curve)\npower_curve['mean'].plot(marker='o', ylabel='Mean power output (kW)', xlabel='Wind-speed interval (m/s)'); plt.xticks(rotation=45); plt.show()"
    notes[76]='Power generally rises from low-wind operation toward higher output; bin counts expose sparsely supported extremes. Pooling different rated-power classes broadens the relationship, so this is not a manufacturer power curve.'
    code[79]="scatterplot(data, 'rotor_speed_rpm', 'generator_speed_rpm')"
    notes[79]='The near-linear rotor/generator relationship indicates strong redundancy. Dispersion around the line is visible, but points should not be labeled faults solely from this plot.'
    code[82]="scatterplot(data, 'wind_speed_mps', 'rotor_speed_rpm')"
    notes[82]='Rotor speed follows wind at low operating speeds and is constrained by turbine controls at higher wind. Nonlinear behavior and dispersion are expected across turbine types.'
    for idx,f in zip([85,88,91,94,97],['drivetrain_vibration_rms_mmps','oil_particle_count','gearbox_bearing_temp_C','power_output_kW','prior_fault_count']): code[idx]=f"boxplot(data, 'failure', {f!r})"
    for idx,key in [(100,'hour'),(103,'dayofweek')]:
        code[idx]=f"rates = data.groupby('{key}').failure.mean().mul(100).sort_index()\ndisplay(rates.rename('failure_pct').to_frame())\nrates.plot.bar(ylabel='Failure rate (%)'); plt.show()\nobservation(f'Highest: {{rates.idxmax()}} ({{rates.max():.3f}}%); lowest: {{rates.idxmin()}} ({{rates.min():.3f}}%). These are descriptive associations, not evidence that calendar time causes faults.')"
    code[103]=code[103].replace("rates.plot.bar", "rates.index = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']\nrates.plot.bar")
    code[107]="numeric_features = data.select_dtypes(include='number').columns.tolist()\nprint(numeric_features)"
    code[109]="corr = data[numeric_features].corr()\nplt.figure(figsize=(16,13)); sns.heatmap(corr, cmap='vlag', center=0); plt.show()\npairs = corr.where(np.triu(np.ones(corr.shape),1).astype(bool)).stack().rename('correlation')\ndisplay(pairs[pairs.abs() >= .9].sort_values())"
    notes[109]='Highly correlated speed and condition channels indicate redundancy. The existing engineered-feature selection is retained for model inputs; correlations describe association and are not causal importance.'
    code[113]="data = data.sort_values(['timestamp','turbine_id'])"
    code[114]="n = len(data)\nprint('Total records:', n)"
    code[115]='''# Preserve the established grouped test and temporal validation protocol.
raw_source = pd.read_csv(root / 'data/raw/wind_turbine_detection.csv', parse_dates=['timestamp'])
metadata = {s: pd.read_csv(root / f'data/processed/03_{s}_metadata.csv', parse_dates=['timestamp']) for s in ['train','validation','test']}
raw_parts = {s: raw_source.merge(m[['turbine_id','timestamp']], on=['turbine_id','timestamp'], validate='one_to_one') for s,m in metadata.items()}
assert not set(metadata['test'].turbine_id) & (set(metadata['train'].turbine_id) | set(metadata['validation'].turbine_id))
for turbine in metadata['train'].turbine_id.unique():
    assert metadata['train'].query('turbine_id == @turbine').timestamp.max() < metadata['validation'].query('turbine_id == @turbine').timestamp.min()
display(pd.DataFrame({s: {'rows': len(p), 'turbines': p.turbine_id.nunique(), 'failure_pct': p.failure.mean()*100} for s,p in raw_parts.items()}).T)
'''
    notes[115]='Adaptation: three entire turbines form the test set; earlier/later records from the other turbines form train/validation. This preserves the existing evaluation design and protects against shared turbine identity in test. It measures unseen-turbine generalization rather than the template’s global future-period split. Earlier test exposure remains a limitation.'
    code[116]='''# Reuse the existing selected snapshot and past-only engineered features.
# Feature-group selection previously used validation data; this reuse is disclosed.
selected_features = pd.read_csv(root / 'reports/model_results/04_selected_features.csv').feature.tolist()
engineered = {s: pd.read_csv(root / f'data/processed/04_{s}_engineered.csv') for s in raw_parts}
X_train, X_valid, X_test = [engineered[s][selected_features].copy() for s in ['train','validation','test']]
y_train, y_valid, y_test = [engineered[s].failure.copy() for s in ['train','validation','test']]
for s, frame in engineered.items():
    assert set(frame.source_index) == set(metadata[s].source_index)
    labels = metadata[s].set_index('source_index').failure.loc[frame.source_index].to_numpy()
    assert np.array_equal(frame.failure, labels)
print('Selected engineered predictors:', selected_features)
'''
    code[121]="display(pd.DataFrame({s:p.isna().mean()*100 for s,p in raw_parts.items()}).rename_axis('feature'))"
    for idx,f in [(124,'gearbox_oil_temp_C'),(127,'generator_bearing_temp_C'),(130,'oil_pressure_bar')]:
        code[idx]=f"fill_value = raw_parts['train'][{f!r}].median()\nfor part in raw_parts.values(): part[{f!r}] = part[{f!r}].fillna(fill_value)\nprint({f!r}, 'training median:', fill_value)\nassert all(p[{f!r}].isna().sum() == 0 for p in raw_parts.values())"
        notes[idx]='Median imputation is robust to extreme values. The same training-derived value is used in every partition. This raw-sensor exercise satisfies the template; the selected engineered predictors below receive their own training-fitted pipeline imputer, including inside cross-validation.'
    code[134]="metric_of_choice = 'recall'\nprint('Primary objective: catch faults (recall); PR-AUC and precision assess false-alert tradeoffs.')"
    code[137]='''def probabilities(model, X):
    if hasattr(model, 'predict_proba'): return model.predict_proba(X)[:,1]
    return model.predict(dataset(np.asarray(X, dtype='float32')), verbose=0).ravel()
def scores(y, p, threshold=.5):
    pred = np.asarray(p) >= threshold
    return dict(Accuracy=accuracy_score(y,pred), Recall=recall_score(y,pred,zero_division=0), Precision=precision_score(y,pred,zero_division=0), F1=f1_score(y,pred,zero_division=0), F2=fbeta_score(y,pred,beta=2,zero_division=0), PR_AUC=average_precision_score(y,p), ROC_AUC=roc_auc_score(y,p))
records, fitted, validation_probabilities = [], {}, {}
def evaluate(name, model, train_x=X_train, valid_x=X_valid):
    fitted[name] = model
    fig, axes = plt.subplots(1,2,figsize=(10,4))
    local = []
    for ax, split, x, y in [(axes[0],'train',train_x,y_train),(axes[1],'validation',valid_x,y_valid)]:
        p = probabilities(model,x)
        row = dict(model=name,split=split,**scores(y,p)); records.append(row); local.append(row)
        ConfusionMatrixDisplay.from_predictions(y,p>=.5,ax=ax,colorbar=False)
        ax.set_title(name+' / '+split)
        if split == 'validation': validation_probabilities[name] = p
    plt.tight_layout(); plt.show(); display(pd.DataFrame(local))
    observation(f"Train/validation recall {local[0]['Recall']:.3f}/{local[1]['Recall']:.3f}, PR-AUC {local[0]['PR_AUC']:.3f}/{local[1]['PR_AUC']:.3f}. A training advantage is evidence of a generalization gap, not proof that tuning will improve validation results.")
def tree_pipeline(model): return Pipeline([('imputer',SimpleImputer(strategy='median')),('model',model)])
'''
    code[138]="def plot_confusion_matrix(model, predictors, target):\n    ConfusionMatrixDisplay.from_predictions(target, probabilities(model,predictors)>=.5); plt.show()"
    specs=[(141,'Decision Tree',"DecisionTreeClassifier(class_weight='balanced', random_state=42)"),(144,'Random Forest',"RandomForestClassifier(n_estimators=150, class_weight='balanced', random_state=42, n_jobs=2)"),(147,'Gradient Boosting',"GradientBoostingClassifier(n_estimators=100, random_state=42)"),(150,'XGBoost',"BalancedXGBClassifier(n_estimators=150,max_depth=6,learning_rate=.1,random_state=42,n_jobs=2,eval_metric='logloss')")]
    for idx,name,expr in specs:
        code[idx]=f"model = tree_pipeline({expr})\nmodel.fit(X_train,y_train)\nevaluate({name!r},model)"
    code[150]="print('Negative / positive training ratio:', (y_train==0).sum()/(y_train==1).sum())\n"+code[150]
    code[154]="ann_transform = Pipeline([('imputer',SimpleImputer(strategy='median')),('scaler',StandardScaler())])\nX_train_scaled = ann_transform.fit_transform(X_train).astype('float32')\nX_valid_scaled = ann_transform.transform(X_valid).astype('float32')\nX_test_scaled = ann_transform.transform(X_test).astype('float32')"
    code[155] = '' # markdown retained
    code[156]="keras.utils.set_random_seed(42)\nmodel = keras.Sequential([keras.layers.Input((X_train_scaled.shape[1],)),keras.layers.Dense(64,activation='relu'),keras.layers.Dense(1,activation='sigmoid')])"
    code[158]="model.summary()"
    code[160]="model.compile(optimizer=keras.optimizers.Adam(.001), loss='binary_crossentropy', metrics=[keras.metrics.Recall(name='recall'),keras.metrics.Precision(name='precision'),keras.metrics.BinaryAccuracy(name='accuracy')])"
    code[162]="epochs = 30\nbatch_size = 128"
    code[163]="weights = compute_class_weight('balanced', classes=np.array([0,1]), y=y_train)\nclass_weight_dict = dict(enumerate(weights))\nsample_weights = np.array([class_weight_dict[int(v)] for v in y_train], dtype='float32')"
    code[164]="start=time.time()\nhistory=model.fit(dataset(X_train_scaled,y_train.to_numpy(),batch_size,sample_weights,True),validation_data=dataset(X_valid_scaled,y_valid.to_numpy()),epochs=epochs,verbose=2,shuffle=False,callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss',patience=5,restore_best_weights=True)])\nend=time.time()"
    code[167]="evaluate('Neural Network',model,X_train_scaled,X_valid_scaled)\nmodel.save(output/'baseline_ann.keras')\npd.DataFrame(history.history).plot(subplots=True,figsize=(10,16)); plt.tight_layout(); plt.show()"
    notes[167]='Metrics are unweighted diagnostic measures; training loss uses class weights and validation loss does not. Early stopping restores the best validation-loss weights. The configured epoch cap satisfies the template, while actual epochs can be fewer.'
    code[171]="baseline_table = pd.DataFrame(records)\ndisplay(baseline_table.query(\"split == 'train'\").sort_values(['Recall','PR_AUC'],ascending=False))"
    code[174]="display(baseline_table.query(\"split == 'validation'\").sort_values(['Recall','PR_AUC'],ascending=False))\nobservation('The top two tree baselines by validation recall, then PR-AUC, are tuned below; the regularized ANN is also evaluated. Selection uses no test metrics.')"
    code[179]='''tree_names = ['Decision Tree','Random Forest','Gradient Boosting','XGBoost']
tuning_candidates = baseline_table.query("split == 'validation' and model in @tree_names").sort_values(['Recall','PR_AUC'],ascending=False).head(2).model.tolist()
print('Selected tree tuning candidates:', tuning_candidates)
# Five expanding chronological folds for XGB/DT/RF; GB uses three as requested.
timestamps = pd.to_datetime(engineered['train'].timestamp)
unique_times = pd.Index(timestamps.unique()).sort_values()
def temporal_folds(count):
    edges = np.linspace(.4,1,count+1); folds=[]
    for left,right in zip(edges[:-1],edges[1:]):
        lo=unique_times[int(left*len(unique_times))]
        hi=unique_times[min(int(right*len(unique_times)),len(unique_times)-1)]
        tr=np.flatnonzero(timestamps<lo)
        va=np.flatnonzero((timestamps>=lo)&((timestamps<hi) if right<1 else (timestamps<=hi)))
        assert timestamps.iloc[tr].max()<timestamps.iloc[va].min()
        folds.append((tr,va))
    return folds
search_spaces = {'XGBoost':{'n_estimators':[100,200], 'learning_rate':[.03,.1], 'max_depth':[3,6], 'min_child_weight':[1,5], 'gamma':[0,.2], 'subsample':[.8,1.], 'colsample_bytree':[.8,1.], 'reg_alpha':[0,.1], 'reg_lambda':[1.,5.]},
'Decision Tree':{'criterion':['gini','entropy'],'max_depth':[5,10,20],'min_samples_split':[2,10],'min_samples_leaf':[1,5,10],'max_features':['sqrt','log2']},
'Random Forest':{'n_estimators':[100,200],'max_depth':[8,16,None],'min_samples_split':[2,10],'min_samples_leaf':[1,5],'max_features':['sqrt','log2']},
'Gradient Boosting':{'n_estimators':[100,150],'learning_rate':[.03,.1],'max_depth':[2,3],'subsample':[.8,1.],'max_features':['sqrt','log2']}}
searches={}
def tune(name):
    if name not in tuning_candidates:
        observation(name+' is not selected for tuning; the template requires at least two selected baseline models.'); return
    folds=temporal_folds(3 if name=='Gradient Boosting' else 5)
    search=RandomizedSearchCV(clone(fitted[name]),{'model__'+k:v for k,v in search_spaces[name].items()},n_iter=4,scoring='recall',cv=folds,refit=True,random_state=42,n_jobs=1,error_score='raise')
    search.fit(X_train,y_train); searches[name]=search
    print('Best parameters:',search.best_params_,'CV recall:',search.best_score_,'folds:',len(folds))
    evaluate(name+' tuned',search.best_estimator_)
display(pd.Series(search_spaces['XGBoost'],name='search_values'))
'''
    code[181]="tune('XGBoost')"
    code[183]="if 'XGBoost' in searches: display(searches['XGBoost'].best_estimator_)\n# RandomizedSearchCV(refit=True) already refits on the complete training set."
    code[187]="keras.utils.set_random_seed(42)\nmodel1 = keras.Sequential([keras.layers.Input((X_train_scaled.shape[1],)),keras.layers.Dense(128,activation='relu'),keras.layers.BatchNormalization(),keras.layers.Dropout(.5),keras.layers.Dense(64,activation='relu'),keras.layers.BatchNormalization(),keras.layers.Dropout(.5),keras.layers.Dense(1,activation='sigmoid')])"
    code[189]="model1.summary()"
    code[191]=code[160].replace('model.compile','model1.compile')
    code[193]="epochs = 30\nbatch_size = 128"
    code[194]=code[164].replace('history=model.fit','history=model1.fit')
    code[197]="evaluate('Neural Network tuned',model1,X_train_scaled,X_valid_scaled)\nmodel1.save(output/'tuned_ann.keras')\npd.DataFrame(history.history).plot(subplots=True,figsize=(10,16)); plt.tight_layout(); plt.show()"
    for name,grid,run,follow,extra in [('Decision Tree',200,202,204,[206,208]),('Random Forest',211,213,215,[]),('Gradient Boosting',218,220,222,[])]:
        code[grid]=f"display(pd.Series(search_spaces[{name!r}],name='search_values'))"
        code[run]=f"tune({name!r})"
        code[follow]=f"if {name!r} in searches: display(searches[{name!r}].best_estimator_)\n# Full-training refit and train/validation evaluation are performed by tune()."
        for idx in extra: code[idx]=f"display(pd.DataFrame(records).query(\"model == '{name} tuned' and split == '{'train' if idx==206 else 'validation'}'\"))"
    code[225]="all_results = pd.DataFrame(records)\ndisplay(all_results.query(\"split == 'train'\").sort_values(['Recall','PR_AUC'],ascending=False))"
    code[227]='''display(all_results.query("split == 'validation'").sort_values(['Recall','PR_AUC'],ascending=False))
for name in tuning_candidates + ['Neural Network']:
    rows=all_results.query("split == 'validation'").set_index('model')
    observation(f"{name}: tuning changes recall by {rows.loc[name+' tuned','Recall']-rows.loc[name,'Recall']:+.4f} and PR-AUC by {rows.loc[name+' tuned','PR_AUC']-rows.loc[name,'PR_AUC']:+.4f} at the same 0.50 threshold. Improvement is assessed from these measured changes, not assumed.")
'''
    code[229]='''ranking=all_results.query("split == 'validation'").sort_values(['Recall','PR_AUC','Precision'],ascending=False)
selected_name=ranking.iloc[0].model
best_model=fitted[selected_name]
is_ann=selected_name.startswith('Neural Network')
final_train=X_train_scaled if is_ann else X_train
final_valid=X_valid_scaled if is_ann else X_valid
final_test=X_test_scaled if is_ann else X_test
# Freeze the common 0.50 decision rule before final test evaluation.
threshold=.5
print('Final model:',selected_name,'selection: validation recall, then PR-AUC and precision; threshold:',threshold)
all_results.to_csv(output/'model_comparison.csv',index=False)
'''
    code[231]='''def plot_feature_importances(model,X,y,feature_names=None):
    if hasattr(model,'named_steps'):
        importance=model.named_steps['model'].feature_importances_
        method='Fitted tree importance'
    else:
        rng=np.random.default_rng(42); indices=rng.choice(len(X),min(3000,len(X)),replace=False)
        sample=np.asarray(X)[indices].copy(); labels=np.asarray(y)[indices]
        base=average_precision_score(labels,probabilities(model,sample)); importance=[]
        for column in range(sample.shape[1]):
            drops=[]
            for repeat in range(3):
                perm=sample.copy(); perm[:,column]=rng.permutation(perm[:,column])
                drops.append(base-average_precision_score(labels,probabilities(model,perm)))
            importance.append(np.mean(drops))
        method='Validation permutation PR-AUC decrease'
    table=pd.DataFrame({'feature':feature_names,'importance':importance}).sort_values('importance',ascending=False)
    display(table.head(20)); table.head(20).sort_values('importance').plot.barh(x='feature',y='importance',figsize=(10,8),title=method); plt.tight_layout(); plt.show()
    observation('Leading input: '+table.iloc[0].feature+'. Importance reflects model association, not causation; correlated predictors can share importance.')
    return table
'''
    code[232]="importance_table=plot_feature_importances(best_model,final_valid,y_valid,selected_features)"
    code[235]='''test_probability=probabilities(best_model,final_test)
final_metrics=scores(y_test,test_probability,threshold)
display(pd.DataFrame([dict(model=selected_name,threshold=threshold,**final_metrics)]))
ConfusionMatrixDisplay.from_predictions(y_test,test_probability>=threshold); plt.title(selected_name+' — held-out test turbines'); plt.show()
tn,fp,fn,tp=confusion_matrix(y_test,test_probability>=threshold).ravel()
print('Detected fault records:',tp,'missed:',fn,'false-positive records:',fp)
(output/'final_metrics.json').write_text(json.dumps(dict(model=selected_name,threshold=threshold,**final_metrics),indent=2))
'''
    notes[238]='The leading model inputs should guide investigation of drivetrain condition, alongside the measured recall/precision tradeoff. False positives consume analyst attention; missed faults can delay intervention. Feature associations do not prove physical causes. The dataset covers 15 turbines over roughly two months, so fleet-wide performance remains uncertain. The current label describes fault detection, not advance warning.'
    notes[240]='Use the classifier for analyst triage and inspection prioritization. Aggregate consecutive positive records into candidate alerts and evaluate cooldown rules before generating work orders. Review missed faults and high-importance sensors with maintenance specialists. Prospectively validate on new turbines and dates, monitor drift and alert burden, and establish an escalation policy before deployment. Previous test exposure and repeated validation reuse limit independence of the reported evaluation.'
    result=nb.v4.new_notebook(metadata={'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'}})
    for idx,cell in enumerate(template.cells):
        cell=copy.deepcopy(cell)
        if idx == 6: cell.source=cell.source.replace('<name>.csv', 'wind_turbine_detection.csv')
        if cell.cell_type=='code':
            cell.source=code.get(idx,cell.source)
            if not cell.source.strip(): raise ValueError(f'Unfilled template code cell {idx}')
            cell.outputs=[];cell.execution_count=None
        elif idx in notes: cell.source=notes.pop(idx)
        cell.metadata['template_cell']=idx
        result.cells.append(cell)
        if idx in notes: result.cells.append(nb.v4.new_markdown_cell('**Observations / implementation notes:** '+notes[idx]))
    result.cells.append(nb.v4.new_markdown_cell('## Additional project analysis\n\nThe original detailed data-quality, feature-engineering, ablation, neural experiments, and operational analysis are preserved in [the archived completed notebook](archive/AIML_Project_1_Full_Code_Notebook_Pre_Template_Alignment.ipynb). Source notebooks 01–08 remain available. This revised submission uses their existing grouped partitions and selected engineered predictors. Its newly executed model results are stored separately in `models/template_aligned/`.'))
    return clean_submission(result, template)

def finalize(notebook):
    # Preserve computed interpretation as actual markdown cells for the rubric.
    expanded=[]
    for cell in notebook.cells:
        expanded.append(cell)
        if cell.cell_type=='code':
            for out in cell.outputs:
                markdown=out.get('data',{}).get('text/markdown')
                if markdown and markdown.startswith('**Observation:**'):
                    expanded.append(nb.v4.new_markdown_cell(markdown))
    notebook.cells=expanded
    import json
    metrics=json.loads((ROOT/'models/template_aligned/final_metrics.json').read_text())
    notebook.cells.append(nb.v4.new_markdown_cell('## Key Findings Recap\n\n' + '\n'.join([
        '- All five required baseline model families were evaluated on training and validation data.',
        '- Two tree families were selected for tuning from baseline validation results; a regularized ANN was also compared.',
        '- The original grouped test and chronological validation partitions were retained.',
        f"- Validation selected {metrics['model']} at the fixed threshold {metrics['threshold']:.2f}.",
        f"- Test recall was {metrics['Recall']:.4f}, precision {metrics['Precision']:.4f}, and accuracy {metrics['Accuracy']:.4f}.",
        '- Earlier test exposure and repeated validation use limit independent confirmation.'
    ])))
    for cell in notebook.cells:
        if cell.metadata.get("template_cell") == 6:
            cell.source=cell.source.replace("<name>.csv", "wind_turbine_detection.csv")
    nb.write(notebook,DEST)

def main():
    notebook=build()
    nb.write(notebook,DEST)
    client=NotebookClient(notebook,timeout=3600,kernel_name='python3',resources={'metadata':{'path':str(ROOT)}}, on_cell_start=lambda cell, cell_index, **kw: print(f'Executing cell {cell_index}', flush=True))
    try:
        client.execute()
    finally:
        nb.write(notebook,DEST)
    if any(o.output_type=='error' for c in notebook.cells if c.cell_type=='code' for o in c.outputs): raise RuntimeError('Execution error')
    finalize(notebook)
    add_model_selection_interpretation(notebook)
    nb.write(notebook,DEST)
    exporter=HTMLExporter(); exporter.embed_images=True
    document,_=exporter.from_notebook_node(notebook)
    (ROOT/'reports/AIML_Project_1_Full_Code_Notebook_Completed.html').write_text(document)
    print('Executed template-aligned notebook and rebuilt HTML.')
if __name__=='__main__': main()
