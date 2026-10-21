"""
Colombo District Land Valuation - XGBoost Model Performance Evaluation
Urban Growth Optimizer IT-337 (Backend Directory Wrapper)
"""
import os
import sys

root_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
eval_script = os.path.join(root_dir, "evaluate_model.py")

with open(eval_script, "r", encoding="utf-8") as f:
    code = f.read()

exec(compile(code, eval_script, "exec"))
