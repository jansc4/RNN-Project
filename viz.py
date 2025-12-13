import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import roc_curve, confusion_matrix, roc_auc_score


def plot_training_history(histories, sensor_names):
    n_sensors = len(histories)
    fig, axes = plt.subplots(1, n_sensors, figsize=(6*n_sensors, 4))
    if n_sensors==1: axes=[axes]
    for ax, (train_losses,val_losses), name in zip(axes,histories,sensor_names):
        ax.plot(train_losses,label='Train Loss'); ax.plot(val_losses,label='Val Loss')
        ax.set_xlabel('Epoch'); ax.set_ylabel('Loss'); ax.legend(); ax.grid(True,alpha=0.3)
        ax.set_title(f'{name.capitalize()} Training')
    plt.tight_layout(); plt.show()


def plot_error_distributions(sensor_errors, sensor_names, thresholds):
    n_sensors = len(sensor_errors)
    fig, axes = plt.subplots(2,n_sensors,figsize=(6*n_sensors,8))
    if n_sensors==1: axes=axes.reshape(-1,1)
    for i, (sensor_name, errors, thresh) in enumerate(zip(sensor_names, sensor_errors, thresholds)):
        normal_errors, anomaly_errors = errors
        normal_errors = normal_errors[np.isfinite(normal_errors)]
        anomaly_errors = anomaly_errors[np.isfinite(anomaly_errors)]
        axes[0,i].hist(normal_errors,bins=50,alpha=0.6,label='Normal',density=True)
        axes[0,i].hist(anomaly_errors,bins=50,alpha=0.6,label='Anomaly',density=True)
        if np.isfinite(thresh): axes[0,i].axvline(thresh,color='r',linestyle='--',label='Threshold')
        axes[0,i].set_title(f'{sensor_name.capitalize()} - Linear'); axes[0,i].legend(); axes[0,i].grid(True,alpha=0.3)
        axes[1,i].hist(normal_errors,bins=50,alpha=0.6,label='Normal',density=True)
        axes[1,i].hist(anomaly_errors,bins=50,alpha=0.6,label='Anomaly',density=True)
        if np.isfinite(thresh): axes[1,i].axvline(thresh,color='r',linestyle='--',label='Threshold')
        axes[1,i].set_title(f'{sensor_name.capitalize()} - Log'); axes[1,i].set_yscale('log'); axes[1,i].legend(); axes[1,i].grid(True,alpha=0.3)
    plt.tight_layout(); plt.show()


def plot_roc_curve(y_true, y_scores, title='ROC Curve'):
    mask=np.isfinite(y_scores)
    if mask.sum()==0: return
    fpr,tpr,_ = roc_curve(y_true[mask], y_scores[mask])
    try: roc_auc=roc_auc_score(y_true[mask], y_scores[mask])
    except: roc_auc=float('nan')
    plt.figure(figsize=(6,6)); plt.plot(fpr,tpr,linewidth=2,label=f'ROC (AUC={roc_auc:.3f})'); plt.plot([0,1],[0,1],'k--',alpha=0.3)
    plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title(title); plt.legend(); plt.grid(True,alpha=0.3); plt.tight_layout(); plt.show()


def plot_confusion_matrix(y_true, y_pred, title='Confusion Matrix'):
    cm=confusion_matrix(y_true,y_pred)
    plt.figure(figsize=(6,5)); sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',cbar=False)
    plt.xlabel('Predicted'); plt.ylabel('Actual'); plt.title(title); plt.tight_layout(); plt.show()