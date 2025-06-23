import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data=pd.read_csv("50_Startups.csv")
data.dropna(inplace=True)

X=data[['R&D Spend', 'Marketing Spend']].values
y=data['Profit'].values

X_mean=np.mean(X, axis=0)
X_std=np.std(X, axis=0)
X_scaled=(X-X_mean)/X_std

np.random.seed(42)
indices=np.random.permutation(len(X_scaled))
train_size=int(0.8*len(X_scaled))
X_train=X_scaled[indices[:train_size]]
X_test=X_scaled[indices[train_size:]]
y_train=y[indices[:train_size]]
y_test=y[indices[train_size:]]

X_train_=np.c_[np.ones(X_train.shape[0]), X_train]
X_test_=np.c_[np.ones(X_test.shape[0]), X_test]
theta=np.linalg.inv(X_train_.T @ X_train_) @ X_train_.T @ y_train
y_pred_lin=X_test_ @ theta

max_depth=5
tree=[]

best_score=float('inf')
for f in range(X_train.shape[1]):
    for t in np.unique(X_train[:, f]):
        left=y_train[X_train[:, f]<=t]
        right=y_train[X_train[:, f]>t]
        if len(left)==0 or len(right)==0:
            continue
        score=len(left)*np.var(left)+len(right)*np.var(right)
        if score<best_score:
            best_score=score
            best_feature=f
            best_threshold=t
            best_left=np.mean(left)
            best_right=np.mean(right)

y_pred_tree=[]
for x in X_test:
    if x[best_feature]<=best_threshold:
        y_pred_tree.append(best_left)
    else:
        y_pred_tree.append(best_right)
y_pred_tree=np.array(y_pred_tree)

C=1.0
epsilon=0.1
alpha=np.zeros(len(X_train))
support_vectors=X_train.copy()

for epoch in range(100):
    for i in range(len(X_train)):
        prediction=np.sum(alpha*(support_vectors @ X_train[i]))
        error=y_train[i]-prediction
        if np.abs(error)>epsilon:
            alpha[i]+=C*error
            alpha[i]=np.clip(alpha[i], 0, C)

y_pred_svr=[]
for x in X_test:
    pred=np.sum(alpha*(support_vectors @ x))
    y_pred_svr.append(pred)
y_pred_svr=np.array(y_pred_svr)

ss_total=np.sum((y_test-np.mean(y_test))**2)

ss_res_lin=np.sum((y_test-y_pred_lin)**2)
r2_lin=1-ss_res_lin/ss_total
mae_lin=np.mean(np.abs(y_test-y_pred_lin))
rmse_lin=np.sqrt(np.mean((y_test-y_pred_lin)**2))

ss_res_tree=np.sum((y_test-y_pred_tree)**2)
r2_tree=1-ss_res_tree/ss_total
mae_tree=np.mean(np.abs(y_test-y_pred_tree))
rmse_tree=np.sqrt(np.mean((y_test-y_pred_tree)**2))

ss_res_svr=np.sum((y_test-y_pred_svr)**2)
r2_svr=1-ss_res_svr/ss_total
mae_svr=np.mean(np.abs(y_test-y_pred_svr))
rmse_svr=np.sqrt(np.mean((y_test-y_pred_svr)**2))

plt.figure(figsize=(18, 6))
models=['Linear Regression', 'Decision Tree', 'SVR']
preds=[y_pred_lin, y_pred_tree, y_pred_svr]
colors=['purple', 'green', 'orange']

for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.scatter(y_test, preds[i], color=colors[i], edgecolor='k', alpha=0.7)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
    plt.xlabel("Actual Profit")
    plt.ylabel("Predicted Profit")
    plt.title(models[i])
    plt.grid(True)

plt.suptitle("Actual vs Predicted Profit", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

r2_scores={
    'Linear Regression': r2_lin,
    'Decision Tree': r2_tree,
    'SVR': r2_svr
}
mae_scores={
    'Linear Regression': mae_lin,
    'Decision Tree': mae_tree,
    'SVR': mae_svr
}
rmse_scores={
    'Linear Regression': rmse_lin,
    'Decision Tree': rmse_tree,
    'SVR': rmse_svr
}

best_model=max(r2_scores, key=r2_scores.get)

print("\nConclusion:")
for model in r2_scores:
    print(f"{model} -> R²: {r2_scores[model]:.4f}, MAE: {mae_scores[model]:.2f}, RMSE: {rmse_scores[model]:.2f}")

print(f"\nBest model is {best_model}, because it has the highest R² value ({r2_scores[best_model]:.4f}),")
print(f"which means it explains the most variance in the data. It also has relatively low MAE and RMSE,")
print("indicating more accurate and consistent predictions.")