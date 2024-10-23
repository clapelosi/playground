import pandas as pd
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st


def chunkify(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def is_axial_orientation(image_orientation):
    # extract points
    A_x, A_y, A_z = float(image_orientation[0]), float(image_orientation[1]), float(image_orientation[2])
    B_x, B_y, B_z = float(image_orientation[3]), float(image_orientation[4]), float(image_orientation[5])
    # if Z asse is close to Origin
    is_axial = (-0.1 < A_z < 0.1) and (-0.1 < B_z < 0.1)
    # if X and Y asses are ortogonal
    is_perpendicular = (A_x * B_x + A_y * B_y + A_z * B_z) < 0.01  # Magari aggiustare la tolleranza

    return is_axial and is_perpendicular


def plot_ROC(points_and_weights, model):
    points=[item['point'] for item in points_and_weights]
    if [0.0, 0.0] not in points: points.append([0.0, 0.0])
    if [1.0, 1.0] not in points: points.append([1.0, 1.0])
    points.sort(key=lambda p: p[0])
    x_vals, y_vals = zip(*points)
    plt.plot(x_vals, y_vals, marker='o', linestyle='-')
    plt.fill_between(x_vals, y_vals, alpha=0.3)
    plt.title(f'ROC Curve {model}')
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.axhline(0, color='black', lw=0.5, ls='--')
    plt.axvline(0, color='black', lw=0.5, ls='--')
    plt.axhline(1, color='black', lw=0.5, ls='--')
    plt.axvline(1, color='black', lw=0.5, ls='--')
    plt.grid()
    # plt.show()
    st.pyplot(plt)
    plt.clf() 


def metrics(df, model_pred, target):
    TP_filter = (df[model_pred]==1) & (df[target]==1)
    FN_filter = (df[model_pred]==0) & (df[target]==1)
    TN_filter = (df[model_pred]==0) & (df[target]==0)
    FP_filter = (df[model_pred]==1) & (df[target]==0)
    TP, FN, TN, FP = len(df[TP_filter]), len(df[FN_filter]), len(df[TN_filter]), len(df[FP_filter])
    TPR = TP / (TP + FN) if (TP + FN) > 0 else 0
    TNR = TN / (TN + FP) if (TN + FP) > 0 else 0
    FPR = 1 - TNR
    return {'TPR': TPR,
            'TNR': TNR,
            'FPR': FPR}


def calculate_thresholds_roc(df, model_pred):
    temp = df.copy()
    df_group = temp.groupby(model_pred).size().reset_index(name='freq')
    freq_tot = df_group['freq'].sum()
    df_group['freq_rel'] = df_group ['freq'] / freq_tot
    df_group = df_group.rename(columns={model_pred:'threshold',
                                        'freq_rel': 'weight'})
    result = df_group[['threshold','weight']].sort_values(by='threshold').to_dict('records')
    return result


def calculate_points_ROC(df, model_pred, target, thresholds_and_weights):
    result = []
    for elem in thresholds_and_weights:
        temp = df.copy()
        temp[model_pred] = temp[model_pred].apply(lambda x: 0 if x < elem['threshold'] else 1)
        TPR = metrics(df=temp, model_pred=model_pred, target=target)['TPR']
        FPR = metrics(df=temp, model_pred=model_pred, target=target)['FPR']
        result.append({'point' : [FPR, TPR],
                    'weight': elem['weight']})
    return result


def calculate_AUROC(points_and_weights):
    points=[item['point'] for item in points_and_weights]
    if [0.0, 0.0] not in points: points.append([0.0, 0.0])
    if [1.0, 1.0] not in points: points.append([1.0, 1.0])
    points.sort(key=lambda p: p[0])
    area = 0
    for i in range(len(points)-1):
        x1, y1 = points[i]
        x2, y2 = points[i + 1]
        area += (x2 - x1) * (y1+ y2) / 2 
    return area


def pick_best_threshold(df, best_model, threshold_list, target, desired_specifity=0.88):
    best_specifity = 0.0
    best_threshold = 0.0
    for threshold in threshold_list:
        temp = df.copy()
        temp[best_model['model']] = temp[best_model['model']].apply(lambda x: 0 if x < threshold else 1)
        specificty = metrics(df=temp, model_pred=best_model['model'], target=target)['TNR']
        st.write(f'threshold {threshold} has a Specifity (True Negative Rate) of: {specificty}\n')
        if abs(best_specifity - desired_specifity) > abs(specificty - desired_specifity):
            best_specifity = specificty
            best_threshold = threshold
    st.write(f'The best threshold is {best_threshold} with a Specifity (True Negative Rate) of: {best_specifity}\n')
    return {'model': best_model['model'],
            'AUROC': best_model['AUROC'],
            'specifity': best_specifity,
            'threshold': best_threshold}
