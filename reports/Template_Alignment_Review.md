# Template alignment verification

- Source: `notebooks/reference/AIML_Project_1_Full_Code_Notebook.ipynb` (copy of the uploaded notebook).
- All 79 analysis headings are preserved in their original order. Assignment boilerplate and task prompts are omitted from the completed submission; observations and executed results demonstrate the requirements.
- All 90 code cells executed in one sequential kernel run without saved errors.
- The notebook and HTML contain 34 plotted outputs, with 34 computed observation markdown cells plus interpretation notes.
- Added all specifically requested EDA plots, tail rows, per-partition missingness, raw-sensor imputation, accuracy, train/validation confusion matrices, neural summaries and training metrics, and baseline/tuned comparison tables.
- Validation selected Gradient Boosting and XGBoost for tree tuning. Gradient Boosting used three chronological folds; XGBoost used five and included gamma and reg_alpha. The template requires at least two tuned baseline models; Decision Tree and Random Forest retain named sections explaining non-selection. A regularized ANN was also trained and evaluated.
- Final validation-recall selection chose the tuned neural network at threshold 0.50. Test accuracy: 0.962735; recall: 0.996616; precision: 0.375398; F1: 0.545370. These replace prior results only in this revised submission.
- Documented adaptations: existing grouped test/temporal validation partitions and 86 previously selected engineered inputs are reused. Raw-sensor imputation is demonstrated separately; actual model pipelines fit their own imputers on training data (inside each tuning fold). Previous test exposure and validation reuse remain disclosed.
- Previous completed notebook preserved in `notebooks/archive/AIML_Project_1_Full_Code_Notebook_Pre_Template_Alignment.ipynb`.
- Existing executive report/site remains the earlier workflow. Updated deliverables are the completed notebook and its matching HTML.

Verification commands: `python scripts/align_template_notebook.py` (full execution); `python scripts/build_full_code_notebook.py` (heading/output validation and HTML export).
