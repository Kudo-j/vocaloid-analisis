import os
import json
import time
import datetime
import copy
import shutil
import ctypes
import optuna

import torch

from train import train_model, evaluate_model

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "util"))
from plot_util.plot import save_loss_plot

# --- Windowsスリープ防止用定数 ---
ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001

def prevent_sleep():
    """Windowsがスリープ状態になるのを阻止する"""
    if os.name == 'nt':
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)

def allow_sleep():
    """Windowsのスリープ阻止を解除する"""
    if os.name == 'nt':
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)


def get_suggested_value(trial, param_name, param_info):
    """JSONの型情報に基づいて、Optunaのtrial.suggestを動的に呼び出す"""
    p_type = param_info.get("type")
    if p_type == "categorical":
        return trial.suggest_categorical(param_name, param_info["choices"])
    elif p_type == "log_float":
        return trial.suggest_float(param_name, param_info["low"], param_info["high"], log=True)
    elif p_type == "float":
        return trial.suggest_float(param_name, param_info["low"], param_info["high"])
    elif p_type == "int":
        return trial.suggest_int(param_name, param_info["low"], param_info["high"])
    else:
        raise ValueError(f"Unknown param type: {p_type}")

class TimeTrackerCallback:
    """各Trial終了時に残り時間と終了予定時刻を予測して表示するコールバック"""
    def __init__(self, total_trials):
        self.total_trials = total_trials
        self.start_time = None
        
    def __call__(self, study, trial):
        if self.start_time is None:
            self.start_time = time.time()
            # 最初の1回は予測不可能なのでスキップ
            return
            
        completed_trials = len(study.trials)
        elapsed_time = time.time() - self.start_time
        
        if completed_trials > 0:
            avg_time_per_trial = elapsed_time / completed_trials
            remaining_trials = self.total_trials - completed_trials
            
            if remaining_trials > 0:
                estimated_remaining_seconds = avg_time_per_trial * remaining_trials
                eta_datetime = datetime.datetime.now() + datetime.timedelta(seconds=estimated_remaining_seconds)
                
                hours, rem = divmod(estimated_remaining_seconds, 3600)
                minutes, _ = divmod(rem, 60)
                
                print(f"\n" + "="*50)
                print(f" [Optuna ETA] Trial {completed_trials}/{self.total_trials} 完了")
                print(f" 残り所要時間: 約 {int(hours)}時間 {int(minutes)}分")
                print(f" 終了予定日時: {eta_datetime.strftime('%Y年%m月%d日 %H:%M 頃')}")
                print("="*50 + "\n")

def main():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # 1. ベース設定と探索空間の読み込み
    with open(os.path.join(BASE_DIR, "train_settings.json"), "r", encoding="utf-8") as f:
        base_config = json.load(f)
        
    with open(os.path.join(BASE_DIR, "search_space.json"), "r", encoding="utf-8") as f:
        search_space = json.load(f)
        
    tune_config = search_space.get("tune_config", {})
    models_to_tune = tune_config.get("MODELS_TO_TUNE", [])
    n_trials = tune_config.get("N_TRIALS_PER_MODEL", 50)
    storage_path = tune_config.get("STORAGE_PATH", "sqlite:///optuna_study.db")
    
    # SQLiteのパスを絶対パス化
    if storage_path.startswith("sqlite:///"):
        db_file = storage_path.replace("sqlite:///", "")
        if not os.path.isabs(db_file):
            storage_path = f"sqlite:///{os.path.join(BASE_DIR, db_file)}"
            
    # 2. モデルごとに独立した探索（Study）を実行
    for model_type in models_to_tune:
        print(f"\n{'*'*60}")
        print(f" Starting Optimization for: {model_type}")
        print(f"{'*'*60}\n")
        
        # Studyの作成（既存のものがあれば続きから再開）
        study_name = f"study_{base_config['EXPERIMENT_NAME']}_{model_type}"
        study = optuna.create_study(
            study_name=study_name,
            storage=storage_path,
            direction="minimize",
            load_if_exists=True
        )
        
        try:
            champion_loss = study.best_value
        except ValueError:
            champion_loss = float('inf')
        
        def objective(trial):
            nonlocal champion_loss
            # ベース設定をコピーして、現在のモデル種別をセット
            config = copy.deepcopy(base_config)
            config["MODEL_TYPE"] = model_type
            
            # --- 共通パラメータのサンプリングと適用 ---
            common_space = search_space.get("common", {})
            for p_name, p_info in common_space.items():
                config[p_name] = get_suggested_value(trial, p_name, p_info)
                
            # --- モデル固有パラメータのサンプリングと適用 ---
            model_space = search_space.get(model_type, {})
            if "MODEL_PARAMS" not in config:
                config["MODEL_PARAMS"] = {}
            if model_type not in config["MODEL_PARAMS"]:
                config["MODEL_PARAMS"][model_type] = {}
                
            for p_name, p_info in model_space.items():
                config["MODEL_PARAMS"][model_type][p_name] = get_suggested_value(trial, p_name, p_info)
                
            # Optuna用に一意な実験名を生成（結果のフォルダが上書きされないようにする）
            config["EXPERIMENT_NAME"] = f"{base_config['EXPERIMENT_NAME']}_optuna"
            config["MODEL_TYPE"] = f"{model_type}_trial_{trial.number}"
            
            # 軽量モードで学習を実行し、best_val_lossと状態を受け取る
            best_val_loss, best_model_state, history_dict = train_model(config, is_tuning=True)
            
            # 過去最高スコア（チャンピオン）を更新した場合のみ保存処理を行う
            if best_val_loss < champion_loss:
                champion_loss = best_val_loss
                
                # 新しいチャンピオンのディレクトリを作成
                EXPERIMENT_DIR = os.path.join(
                    BASE_DIR, 
                    config.get("OUTPUT_BASE_DIR", "output"), 
                    config["EXPERIMENT_NAME"], 
                    config["MODEL_TYPE"]
                )
                os.makedirs(EXPERIMENT_DIR, exist_ok=True)
                
                # ベストモデルの保存
                model_save_path = os.path.join(EXPERIMENT_DIR, "best_model.pth")
                torch.save(best_model_state, model_save_path)
                
                # 学習曲線の保存
                plot_save_path = os.path.join(EXPERIMENT_DIR, "loss_curve.png")
                save_loss_plot(
                    history_dict["train_loss"], history_dict["val_loss"],
                    history_dict["train_metrics"], history_dict["val_metrics"],
                    plot_save_path, history_dict["best_epoch"], best_val_loss, history_dict["hyperparams"]
                )

                # 推論用スナップショットの保存 (train_config.json & features.json)
                train_config_save_path = os.path.join(EXPERIMENT_DIR, "train_config.json")
                with open(train_config_save_path, "w", encoding="utf-8") as f:
                    json.dump(config, f, indent=4, ensure_ascii=False)
                
                features_src = os.path.join(config["DATA_DIR"], "features.json")
                if os.path.exists(features_src):
                    shutil.copyfile(features_src, os.path.join(EXPERIMENT_DIR, "features.json"))
            
            return best_val_loss

        # 最適化の実行（コールバックを渡す）
        time_tracker = TimeTrackerCallback(n_trials)
        study.optimize(objective, n_trials=n_trials, callbacks=[time_tracker])
        
        # 探索完了後の結果出力
        print(f"\nOptimization finished for {model_type}.")
        best_trial_num = study.best_trial.number
        print(f"Best Trial: {best_trial_num}")
        print(f"Best Value (Val Loss): {study.best_value:.6f}")
        
        # --- 最強モデルでの最終評価 (Evaluate) ---
        best_config = copy.deepcopy(base_config)
        best_config["MODEL_TYPE"] = f"{model_type}_trial_{best_trial_num}"
        best_config["EXPERIMENT_NAME"] = f"{base_config['EXPERIMENT_NAME']}_optuna"
        
        # ベストパラメータを復元
        for p_name in search_space.get("common", {}):
            if p_name in study.best_trial.params:
                best_config[p_name] = study.best_trial.params[p_name]
                
        if "MODEL_PARAMS" not in best_config:
            best_config["MODEL_PARAMS"] = {}
        if model_type not in best_config["MODEL_PARAMS"]:
            best_config["MODEL_PARAMS"][model_type] = {}
        for p_name in search_space.get(model_type, {}):
            if p_name in study.best_trial.params:
                best_config["MODEL_PARAMS"][model_type][p_name] = study.best_trial.params[p_name]
                
        # Evaluator実行
        EXPERIMENT_DIR = os.path.join(
            BASE_DIR, 
            best_config.get("OUTPUT_BASE_DIR", "output"), 
            best_config["EXPERIMENT_NAME"], 
            best_config["MODEL_TYPE"]
        )
        model_load_path = os.path.join(EXPERIMENT_DIR, "best_model.pth")
        
        if os.path.exists(model_load_path):
            print("\n>>> Running Final Evaluation on the Champion Model <<<")
            evaluate_model(best_config, model_load_path)
        else:
            print(f"Warning: Champion model file not found at {model_load_path}")
        
        # テキストファイルとしてベストパラメータを出力
        output_dir = os.path.join(BASE_DIR, base_config.get("OUTPUT_BASE_DIR", "output"))
        os.makedirs(output_dir, exist_ok=True)
        best_params_path = os.path.join(output_dir, f"best_params_{model_type}.txt")
        
        print("Best Params:")
        with open(best_params_path, "w", encoding="utf-8") as f:
            f.write(f"Best Trial: {study.best_trial.number}\n")
            f.write(f"Best Value (Val Loss): {study.best_value:.6f}\n")
            f.write("Best Params:\n")
            for key, value in study.best_trial.params.items():
                f.write(f'    "{key}": {value},\n')
                print(f"    {key}: {value}")
                
        print(f"-> Best parameters saved to {best_params_path}")

if __name__ == "__main__":
    prevent_sleep()
    try:
        main()
    finally:
        allow_sleep()
