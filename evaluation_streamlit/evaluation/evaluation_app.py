import pandas as pd
import streamlit as st
from concurrent.futures import ProcessPoolExecutor
import utils as uu


def find_false_negative_patients(df, thresholds, desired_specifity):

    # params
    target = 'Target'
    id_patient = 'ID'
    N_WORKERS = 4

    # EVALUATIONS OF THE PREDICTIONS OF DIFFERENT MODELS 
    models_pred_cols = [col for col in df.columns if 'Model' in col]
    results = []
    for col in models_pred_cols:
        st.write(f"### Start Evalutation for {col} ###")
        
        thresholds_and_weights = uu.calculate_thresholds_roc(df=df.copy(), model_pred=col)
        points_and_weights = []
        with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
            futures = [executor.submit(uu.calculate_points_ROC, df=df, model_pred=col,
                                       target=target, thresholds_and_weights=chunk)
                       for chunk in uu.chunkify(thresholds_and_weights, N_WORKERS)]
            for future in futures:
                points_and_weights.extend(future.result())

        # COLLECT RESULTS
        AUROC = uu.calculate_AUROC(points_and_weights=points_and_weights)

        # Append results for display
        results.append({
            'model': col,
            'AUROC': AUROC,
            'points_and_weights': points_and_weights,
        })

    # Display results in expanders
    for result in results:
        with st.expander(f"Detail for: {result['model']} (AUROC: {result['AUROC']})"):
            uu.plot_ROC(result['points_and_weights'], result['model'])

    # COLLECT THE BEST MODEL NAME AND ITS AUROC  
    best_model = max(results, key=lambda x: x['AUROC'])
    st.write(f"### The best model is: {best_model['model']} with AUROC of: {best_model['AUROC']} ###")

    # CHOOSING THE BEST GIVEN THRESHOLD
    st.write('### Start of Evaluation of threshold ###')
    best_model_spec = uu.pick_best_threshold(df=df, best_model=best_model, threshold_list=thresholds,
                                             target=target, desired_specifity=desired_specifity)
    st.write("### End of Evaluation threshold ###")
    
    # FINDING THE FALSE NEGATIVE PATIENTS
    temp = df.copy()
    temp[best_model_spec['model']] = temp[best_model_spec['model']].apply(lambda x: 0 if x < best_model_spec['threshold'] else 1)
    FN_filter = (temp[best_model_spec['model']] == 0) & (temp[target] == 1)
    FN_patients = temp[FN_filter][id_patient].tolist()
    
    st.write('### List of the False Negative Patients ###')
    st.write(FN_patients)
    st.write(f"Calculated with the model {best_model_spec['model']} of AUROC: {best_model_spec['AUROC']}")
    st.write(f"Considered threshold: {best_model_spec['threshold']} => Specifity of: {best_model_spec['specifity']}.")


def main():
    st.title("False Negative Patiants Evaluation")
    
    # File Uploader to upload the csv
    uploaded_file = st.file_uploader("Upload Model Results csv file", type="csv")
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("### Data Uploaded ###")
        st.write(df.head())  # Show first rows

        threshold_input = st.text_input("Isert the thresholds List (comma separated):", "0.3, 0.5, 0.7, 0.9")
        
        # Transform the input string into a list of float
        try:
            thresholds = [float(t.strip()) for t in threshold_input.split(",")]
        except ValueError:
            st.error("Error: Be sure to instert only comma separated values.")
            return

        desired_specifity = st.number_input("Insert your desired Specifity:", 0.0, 1.0, 0.8)
        
        if st.button("Execute Evaluation"):
            find_false_negative_patients(df, thresholds, desired_specifity)


if __name__ == "__main__":
    main()
