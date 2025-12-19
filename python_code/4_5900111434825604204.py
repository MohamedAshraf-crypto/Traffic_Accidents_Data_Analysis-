# traffic_gui.py
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from PIL import Image, ImageTk
import customtkinter as ctk
import tkinter as tk

# -------------------- Settings --------------------
BG_IMAGE_PATH = "Conduire sans risque.jpeg"  
DATA_CSV = "data_merge.csv"                  
MODEL_FILE = "traffic_model.joblib"

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# -------------------- Utility --------------------
def load_or_create_data():
    if Path(DATA_CSV).exists():
        df = pd.read_csv(DATA_CSV)
    else:
        rng = np.random.default_rng(42)
        n = 100
        df = pd.DataFrame({
            "vehicle_type": rng.choice(["Car", "Truck", "Motorbike"], size=n),
            "age_of_driver": rng.integers(18,80,size=n),
            "casualty_severity": rng.choice([0,1], size=n)
        })
    return df

def prepare_pipeline(df, target_col="casualty_severity"):
    features = [c for c in df.columns if c != target_col]
    X = df[features].copy()
    y = df[target_col].copy()

    X = X.fillna(0)
    y = y.fillna(y.mode()[0] if not y.isnull().all() else 0)

    X_enc = pd.get_dummies(X, drop_first=False)
    return X_enc, y, features

def train_and_save_model(df):
    X_enc, y, features = prepare_pipeline(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X_enc, y, test_size=0.2, random_state=42, stratify=y if len(np.unique(y))>1 else None
    )
    model = RandomForestClassifier(n_estimators=150, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    joblib.dump({"model": model, "columns": X_enc.columns.tolist(), "features": features}, MODEL_FILE)
    return model, X_enc.columns.tolist(), features, acc

# -------------------- Load data & model --------------------
df = load_or_create_data()
try:
    if Path(MODEL_FILE).exists():
        data = joblib.load(MODEL_FILE)
        model = data["model"]
        trained_columns = data["columns"]
        features = data["features"]
    else:
        model, trained_columns, features, acc = train_and_save_model(df)
except Exception:
    model, trained_columns, features, acc = train_and_save_model(df)

# -------------------- GUI --------------------
class TrafficApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Traffic Accident Predictor - سوو")
        self.geometry("980x640")
        self.resizable(True, True)

        # background image
        self._set_background(BG_IMAGE_PATH)

        # main frame
        self.frame = ctk.CTkFrame(self, fg_color="#0b1220", corner_radius=12)
        self.frame.place(relx=0.02, rely=0.02, relwidth=0.96, relheight=0.96)

        # Title
        title = ctk.CTkLabel(
            self.frame, text="🚦 Traffic Accident Predictor",
            font=ctk.CTkFont(size=22, weight="bold"), text_color="#ff7b7b"
        )
        title.pack(pady=(10,6))

        sub = ctk.CTkLabel(
            self.frame, text="ادخلي بيانات الحالة واظغطي Predict عشان تشوفي النتيجة",
            font=ctk.CTkFont(size=12), text_color="#cbd8e6"
        )
        sub.pack(pady=(0,8))

        # scrollable frame for form
        self.form = ctk.CTkScrollableFrame(self.frame, fg_color="#071022", corner_radius=8)
        self.form.pack(padx=20, pady=10, fill="both", expand=True)

        self.feature_widgets = {}
        show_order = [f for f in df.columns if f != "casualty_severity"]

        for f in show_order:
            lbl = ctk.CTkLabel(self.form, text=f, anchor="w")
            lbl.pack(padx=12, pady=(6,0), fill="x")
            if df[f].dtype == "object" or str(df[f].dtype).startswith("category"):
                opts = sorted(df[f].dropna().unique().tolist())
                if not opts:
                    opts = ["N/A"]
                var = ctk.StringVar(value=opts[0])
                widget = ctk.CTkOptionMenu(self.form, values=opts, variable=var)
                widget.pack(padx=12, pady=(2,6), fill="x")
                self.feature_widgets[f] = ("cat", var)
            else:
                default = float(df[f].dropna().mean()) if not df[f].dropna().empty else 0.0
                var = tk.StringVar(value=str(round(default, 2)))
                widget = ctk.CTkEntry(self.form, textvariable=var)
                widget.pack(padx=12, pady=(2,6), fill="x")
                self.feature_widgets[f] = ("num", var)

        # predict button
        self.btn = ctk.CTkButton(self.frame, text="🔮 Predict", command=self.on_predict, width=180)
        self.btn.pack(pady=(6,10))

        # results area
        self.result_label = ctk.CTkLabel(self.frame, text="", font=ctk.CTkFont(size=16, weight="bold"))
        self.result_label.pack(pady=(2,10))

        # info
        info_txt = f"Model trained on {'your' if Path(DATA_CSV).exists() else 'synthetic'} data. Model file: {MODEL_FILE}"
        info = ctk.CTkLabel(self.frame, text=info_txt, text_color="#cbd8e6", font=ctk.CTkFont(size=10))
        info.pack(pady=(2,6))

    def _set_background(self, image_path):
        if Path(image_path).exists():
            try:
                img = Image.open(image_path).convert("RGBA")
                img = img.resize((980, 640), Image.LANCZOS)
                self.bg_img = ImageTk.PhotoImage(img)
                bg_label = tk.Label(self, image=self.bg_img)
                bg_label.place(x=0, y=0, relwidth=1, relheight=1)
            except Exception as e:
                print("Background load failed:", e)
        else:
            self.configure(bg="#0b1220")

    def on_predict(self):
        rec = {}
        for f, (ftype, var) in self.feature_widgets.items():
            val = var.get()
            if ftype == "num":
                try:
                    rec[f] = float(val)
                except:
                    self.result_label.configure(text="⚠️ Please enter valid numeric values", text_color="#ff6b6b")
                    return
            else:
                rec[f] = val
        input_df = pd.DataFrame([rec])
        input_enc = pd.get_dummies(input_df)
        X_input = input_enc.reindex(columns=trained_columns, fill_value=0)
        try:
            pred = model.predict(X_input)[0]
            if pred == 1:
                txt = "💀 Predicted: يموت"
                color = "#ff4d4d"
            else:
                txt = "😊 Predicted: يعيش"
                color = "#7ef9b6"
            self.result_label.configure(text=txt, text_color=color)
        except Exception as e:
            self.result_label.configure(text=f"Error predicting: {e}", text_color="#ff6b6b")


if __name__ == "__main__":
    app = TrafficApp()
    app.mainloop()
