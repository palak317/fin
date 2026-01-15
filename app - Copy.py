import os
import re
import json
import time
import pandas as pd
import google.generativeai as genai
import yfinance as yf  # Added for Wealth Oracle real-time prices
from flask import Flask, render_template, request, jsonify
from twilio.twiml.messaging_response import MessagingResponse
from io import StringIO

app = Flask(__name__)
app.secret_key = "cyberpunk_secret_key"
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# --- CONFIGURATION ---
GEMINI_API_KEY = ""  # Replace with your working key
genai.configure(api_key=GEMINI_API_KEY)

# --- GLOBAL DATA STORE ---
transactions = []  # For Live Tracker
bank_data_store = []  # For Neural Analyzer Context


# --- HELPER: Live Price ---
def get_live_price(ticker):
    try:
        if not ticker: return "N/A"
        if not ticker.endswith('.NS') and not ticker.endswith('.BO'): ticker += '.NS'
        stock = yf.Ticker(ticker)
        price = stock.fast_info.get('last_price', None)
        return round(price, 2) if price else "N/A"
    except:
        return "N/A"


# --- YOUR ORIGINAL MODEL HOPPER ---
def generate_with_fallback(file_obj, prompt):
    candidates = [
        'gemini-1.5-flash', 'gemini-flash-latest', 'gemini-2.0-flash-lite-preview-02-05',
        'gemini-pro', 'gemini-1.5-pro-latest'
    ]
    last_error = ""
    for model_name in candidates:
        try:
            model = genai.GenerativeModel(model_name)
            if file_obj:
                response = model.generate_content([file_obj, prompt])
            else:
                response = model.generate_content(prompt)
            return response
        except Exception as e:
            last_error = str(e)
            continue
    raise Exception(f"All models failed. Last error: {last_error}")


# ==========================================
#  STEP 1: EXTRACT RAW DATA (YOUR CODE)
# ==========================================
@app.route('/api/analyze_bank', methods=['POST'])
def analyze_bank():
    if 'file' not in request.files: return jsonify({"error": "No file uploaded"}), 400
    file = request.files['file']
    if file.filename == '': return jsonify({"error": "No file selected"}), 400

    temp_path = os.path.join(app.config['UPLOAD_FOLDER'], "temp.pdf")
    try:
        file.save(temp_path)
        g_file = genai.upload_file(path=temp_path, display_name="Bank Statement")

        # Wait loop (Your logic)
        for _ in range(10):
            if g_file.state.name == "ACTIVE": break
            time.sleep(1)
            g_file = genai.get_file(g_file.name)

        if g_file.state.name == "FAILED": raise Exception("Google PDF processing failed.")

        prompt = """
        Extract transaction table. Output ONLY raw CSV format.
        Columns: Date, Description, Withdrawal, Deposit.
        (If transaction is an expense, put value in Withdrawal. If income, put in Deposit. Use 0 for empty).
        Clean data, merge descriptions.
        Format numbers purely (e.g. 1500.00).
        """

        response = generate_with_fallback(g_file, prompt)

        try:
            genai.delete_file(g_file.name)
            os.remove(temp_path)
        except:
            pass

        csv_text = response.text.replace("```csv", "").replace("```", "").strip()
        df = pd.read_csv(StringIO(csv_text))
        df = df.fillna(0)

        return jsonify({
            "status": "success",
            "columns": df.columns.tolist(),
            "data": df.to_dict(orient='records')
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ==========================================
#  STEP 2: SAVE & GENERATE CHARTS (YOUR CODE)
# ==========================================
@app.route('/api/save_transactions', methods=['POST'])
def save_transactions():
    try:
        data = request.json.get('data')
        global bank_data_store
        bank_data_store = data  # Store for context if needed

        df = pd.DataFrame(data)

        # 1. Force convert columns to Numbers
        df['Withdrawal'] = pd.to_numeric(df['Withdrawal'], errors='coerce').fillna(0.0)
        df['Deposit'] = pd.to_numeric(df['Deposit'], errors='coerce').fillna(0.0)

        income = float(df['Deposit'].sum())
        expense = float(df['Withdrawal'].sum())
        net = float(income - expense)

        # --- PREPARE DATA FOR 6 CHARTS ---
        cat_series = df[df['Withdrawal'] > 0].groupby('Category')['Withdrawal'].sum()
        cat_data = {k: float(v) for k, v in cat_series.items()}

        trend_exp_series = df.groupby('Date')['Withdrawal'].sum()
        trend_exp_data = {str(k): float(v) for k, v in trend_exp_series.items()}

        top_exp_df = df.nlargest(5, 'Withdrawal')[['Description', 'Withdrawal']]
        top_exp_data = {row['Description'][:15] + '..': float(row['Withdrawal']) for _, row in top_exp_df.iterrows()}

        freq_series = df[df['Withdrawal'] > 0]['Category'].value_counts()
        freq_data = {k: int(v) for k, v in freq_series.items()}

        trend_inc_series = df.groupby('Date')['Deposit'].sum()
        trend_inc_data = {str(k): float(v) for k, v in trend_inc_series.items()}

        return jsonify({
            "status": "success",
            "stats": {
                "income": round(income, 2),
                "expense": round(expense, 2),
                "net": round(net, 2),
                "charts": {
                    "cat_data": cat_data,
                    "trend_exp": trend_exp_data,
                    "top_exp": top_exp_data,
                    "freq_data": freq_data,
                    "trend_inc": trend_inc_data
                }
            }
        })
    except Exception as e:
        print(f"Server Error in Save: {e}")
        return jsonify({"error": str(e)}), 500


# ==========================================
#  NEW FEATURE: AI EXPENSE INSIGHT
# ==========================================
@app.route('/api/generate_insight', methods=['POST'])
def generate_insight():
    try:
        stats = request.json.get('stats')
        prompt = f"""
        Act as a financial auditor. Analyze this spending summary:
        Income: {stats['income']}, Expense: {stats['expense']}, Net: {stats['net']}.
        Top Categories: {stats['charts']['cat_data']}.

        Write a short, sharp paragraph (HTML format) analyzing the user's financial health. 
        Use <b> tags for key points. Be direct.
        """
        response = generate_with_fallback(None, prompt)
        return jsonify({"insight": response.text})
    except Exception as e:
        return jsonify({"insight": "AI could not generate insight at this moment."})


# ==========================================
#  WEALTH ORACLE (ENHANCED FOR UI)
# ==========================================
@app.route('/api/advisor', methods=['POST'])
def advisor():
    data = request.json
    try:
        # Improved prompt to return the specific JSON structure the UI needs
        prompt = f"""
        Act as a Hedge Fund Manager. 
        User Profile: Age {data.get('age')}, Risk {data.get('risk')}, Job {data.get('job')}, Capital {data.get('amount')}.

        Generate an investment strategy.
        Output ONLY valid JSON in this exact format:
        {{
            "market_sentiment": "Bullish/Bearish/Neutral",
            "strategy_note": "A 2-sentence summary of the strategy.",
            "portfolio": [
                {{ "asset": "Stock Name", "ticker": "RELIANCE.NS", "alloc": "20%", "rationale": "Why?" }},
                {{ "asset": "Stock Name 2", "ticker": "TCS.NS", "alloc": "15%", "rationale": "Why?" }}
            ]
        }}
        """
        response = generate_with_fallback(None, prompt)

        # Clean response
        clean_text = response.text.replace("```json", "").replace("```", "").strip()
        advice = json.loads(clean_text)

        # Fetch Live Prices (Safe Mode)
        for item in advice.get('portfolio', []):
            item['live_price'] = get_live_price(item.get('ticker'))

        return jsonify(advice)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ==========================================
#  LIVE TRACKER (YOUR WEBHOOK + GET)
# ==========================================
@app.route('/webhook', methods=['POST'])
def webhook():
    msg = request.values.get('Body', '').lower()
    resp = MessagingResponse()
    d_str = time.strftime("%Y-%m-%d")

    if 'received' in msg or 'credit' in msg:
        amt = re.findall(r'\d+', msg)
        if amt:
            transactions.append({
                'Date': d_str,
                'Description': 'Income',
                'Category': 'Income',
                'Withdrawal': 0,
                'Deposit': float(amt[0])
            })
            resp.message(f"✅ Credit: {amt[0]}")
    else:
        amt = re.search(r'(\d+)', msg)
        if amt:
            desc = re.sub(r'\d+', '', msg).strip() or "Expense"
            transactions.append({
                'Date': d_str,
                'Description': desc,
                'Category': 'Expense',
                'Withdrawal': float(amt.group(1)),
                'Deposit': 0
            })
            resp.message(f"📉 Debit: {amt.group(1)}")

    return str(resp)


@app.route('/api/tracker_data', methods=['GET'])
def tracker_data():
    # Return transactions + calculated stats for the tracker chart
    if not transactions:
        return jsonify({"transactions": [], "stats": {"income": 0, "expense": 0, "net": 0}})

    df = pd.DataFrame(transactions)
    income = float(df['Deposit'].sum()) if 'Deposit' in df else 0
    expense = float(df['Withdrawal'].sum()) if 'Withdrawal' in df else 0

    return jsonify({
        "transactions": list(reversed(transactions)),
        "stats": {
            "income": income,
            "expense": expense,
            "net": income - expense
        }
    })


@app.route('/')
def home():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True, port=5000)