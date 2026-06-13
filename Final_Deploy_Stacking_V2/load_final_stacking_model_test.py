
import cloudpickle
import joblib
import numpy as np
import pandas as pd

SAVE_DIR = "Final_Deploy_Stacking_V2"

# 读取完整部署包
with open(f"{SAVE_DIR}/Final_Raw_Stacking_Deploy_Pack_V2_cloudpickle.pkl", "rb") as f:
    deploy_pack = cloudpickle.load(f)

model = deploy_pack["final_deploy_model"]
feature_order = deploy_pack["feature_order"]

X_test = deploy_pack["datasets"]["X_test_final"]

# 确保特征顺序一致
X_test = X_test[feature_order]

y_prob = model.predict_proba(X_test)[:, 1]

print("✅ 模型读取成功。")
print("前10个预测概率：")
print(y_prob[:10])
