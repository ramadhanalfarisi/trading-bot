//+------------------------------------------------------------------+
//|  MLAdvisorONNX.mq5                                              |
//|  Expert Advisor — Inferensi langsung menggunakan model ONNX     |
//|  tanpa memerlukan server Python terpisah.                        |
//|                                                                  |
//|  Prerequisite:                                                   |
//|  - Copy file forex_advisor.onnx ke folder MQL5/Files/           |
//|  - MT5 build >= 3450 (ONNX support)                             |
//+------------------------------------------------------------------+
#property copyright "ML Forex Advisor ONNX"
#property version   "1.00"

#include <Trade\Trade.mqh>
#include <Trade\SymbolInfo.mqh>

//--- Input parameters
input string   OnnxModelFile   = "forex_advisor.onnx"; // File ONNX (di MQL5/Files/)
input int      SeqLength       = 60;                    // Panjang sequence (harus sama dengan training)
input int      NumFeatures     = 80;                    // Jumlah fitur (lihat log training)
input double   ConfThreshold   = 0.60;                  // Min confidence
input double   LotSize         = 0.01;                  // Lot size
input double   SL_ATR_Mult     = 2.0;                   // SL multiplier ATR
input double   TP_ATR_Mult     = 3.0;                   // TP multiplier ATR
input int      ATR_Period      = 14;                    // ATR period
input int      MagicNumber     = 20240102;              // Magic number
input bool     EnableTrading   = true;                  // Aktifkan trading

//--- Global
CTrade         g_trade;
CSymbolInfo    g_symbol;
long           g_onnx_handle = INVALID_HANDLE;
datetime       g_last_bar   = 0;
int            g_atr_handle = INVALID_HANDLE;

// Label mapping: 0=HOLD, 1=BUY, 2=SELL
string LABELS[] = {"HOLD", "BUY", "SELL"};

//+------------------------------------------------------------------+
//| Init                                                             |
//+------------------------------------------------------------------+
int OnInit()
{
   g_symbol.Name(_Symbol);
   g_trade.SetExpertMagicNumber(MagicNumber);

   // Muat ONNX model
   g_onnx_handle = OnnxCreate(OnnxModelFile, ONNX_DEFAULT);
   if(g_onnx_handle == INVALID_HANDLE)
   {
      Print("ERROR: Gagal memuat ONNX model '", OnnxModelFile, "'. Error: ", GetLastError());
      Print("Pastikan file ada di folder MQL5/Files/");
      return INIT_FAILED;
   }

   // Set shape input: [1, SeqLength, NumFeatures]
   ulong input_shape[] = {1, (ulong)SeqLength, (ulong)NumFeatures};
   if(!OnnxSetInputShape(g_onnx_handle, 0, input_shape))
   {
      Print("ERROR: Gagal set input shape: ", GetLastError());
      return INIT_FAILED;
   }

   // Set shape output: [1, 3]
   ulong output_shape[] = {1, 3};
   if(!OnnxSetOutputShape(g_onnx_handle, 0, output_shape))
   {
      Print("ERROR: Gagal set output shape: ", GetLastError());
      return INIT_FAILED;
   }

   // Inisialisasi indikator ATR
   g_atr_handle = iATR(_Symbol, PERIOD_CURRENT, ATR_Period);
   if(g_atr_handle == INVALID_HANDLE)
   {
      Print("ERROR: Gagal inisialisasi ATR");
      return INIT_FAILED;
   }

   Print("✅ ONNX model dimuat | Seq: ", SeqLength, " | Features: ", NumFeatures);
   return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| Deinit                                                           |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   if(g_onnx_handle != INVALID_HANDLE) OnnxRelease(g_onnx_handle);
   if(g_atr_handle  != INVALID_HANDLE) IndicatorRelease(g_atr_handle);
}

//+------------------------------------------------------------------+
//| Tick                                                             |
//+------------------------------------------------------------------+
void OnTick()
{
   datetime current_bar = iTime(_Symbol, PERIOD_CURRENT, 0);
   if(current_bar == g_last_bar) return;
   g_last_bar = current_bar;

   // Build feature tensor
   float input_tensor[];
   if(!BuildFeatureTensor(input_tensor))
   {
      Print("Gagal build feature tensor");
      return;
   }

   // Run ONNX inference
   float output_tensor[];
   ArrayResize(output_tensor, 3);

   if(!OnnxRun(g_onnx_handle,
               ONNX_DEBUG_LOGS,
               input_tensor,
               output_tensor))
   {
      Print("ERROR: ONNX inference gagal: ", GetLastError());
      return;
   }

   // Softmax (model mungkin output logits)
   float proba[];
   Softmax(output_tensor, proba);

   // Tentukan kelas terbaik
   int best_class = 0;
   float best_conf = proba[0];
   for(int i = 1; i < 3; i++)
   {
      if(proba[i] > best_conf)
      {
         best_conf  = proba[i];
         best_class = i;
      }
   }

   // Override ke HOLD jika confidence rendah
   if(best_conf < ConfThreshold) best_class = 0;

   string signal = LABELS[best_class];
   Print(StringFormat("Signal: %s | HOLD:%.3f BUY:%.3f SELL:%.3f",
         signal, proba[0], proba[1], proba[2]));

   // Hitung ATR untuk SL/TP
   double atr_buf[];
   CopyBuffer(g_atr_handle, 0, 1, 1, atr_buf);
   double atr = atr_buf[0];
   double sl_dist = atr * SL_ATR_Mult;
   double tp_dist = atr * TP_ATR_Mult;

   double bid = g_symbol.Bid();
   double ask = g_symbol.Ask();

   Comment(StringFormat("ONNX Advisor | %s | Conf: %.1f%% | ATR: %.5f",
           signal, best_conf * 100, atr));

   if(!EnableTrading) return;

   if(signal == "BUY")
   {
      CloseByType(POSITION_TYPE_SELL);
      if(!HasPosition(POSITION_TYPE_BUY))
         g_trade.Buy(LotSize, _Symbol, ask, ask - sl_dist, ask + tp_dist, "ML-ONNX-BUY");
   }
   else if(signal == "SELL")
   {
      CloseByType(POSITION_TYPE_BUY);
      if(!HasPosition(POSITION_TYPE_SELL))
         g_trade.Sell(LotSize, _Symbol, bid, bid + sl_dist, bid - tp_dist, "ML-ONNX-SELL");
   }
}

//+------------------------------------------------------------------+
//| Build feature tensor — SESUAIKAN dengan feature_engineering.py  |
//| Urutan fitur HARUS SAMA dengan yang digunakan saat training.    |
//| Lihat: preprocessor.feature_cols setelah training selesai.      |
//+------------------------------------------------------------------+
bool BuildFeatureTensor(float &tensor[])
{
   int total_size = SeqLength * NumFeatures;
   ArrayResize(tensor, total_size);
   ArrayInitialize(tensor, 0.0);

   MqlRates rates[];
   int copied = CopyRates(_Symbol, PERIOD_CURRENT, 0, SeqLength + 50, rates);
   if(copied < SeqLength)
   {
      Print("Data tidak cukup: ", copied, " bar");
      return false;
   }

   // Contoh: isi fitur sederhana (sesuaikan dengan feature engineering Python)
   // Anda perlu mereplikasi PERSIS logika FeatureEngineer.build_features() di sini
   // atau gunakan MLAdvisor.mq5 (versi API) yang lebih mudah dipelihara.
   for(int t = 0; t < SeqLength; t++)
   {
      int r = copied - SeqLength + t;  // index rates
      int base = t * NumFeatures;

      // Fitur dasar (normalisasi sederhana menggunakan recent mean/std)
      // CATATAN: Untuk produksi, gunakan nilai mean/std dari training (simpan di file)
      double close  = rates[r].close;
      double open   = rates[r].open;
      double high   = rates[r].high;
      double low    = rates[r].low;
      double vol    = (double)rates[r].tick_volume;

      // Return features
      double ret1 = (r > 0) ? (close - rates[r-1].close) / rates[r-1].close : 0;
      double hl   = (high - low) / close;
      double oc   = (close - open) / open;

      tensor[base + 0] = (float)ret1;
      tensor[base + 1] = (float)hl;
      tensor[base + 2] = (float)oc;

      // Sisa fitur diisi 0 sampai pipeline normalisasi penuh diimplementasikan
      // Untuk implementasi produksi lengkap, gunakan versi API (MLAdvisor.mq5)
   }
   return true;
}

//+------------------------------------------------------------------+
//| Softmax                                                          |
//+------------------------------------------------------------------+
void Softmax(const float &logits[], float &proba[])
{
   ArrayResize(proba, ArraySize(logits));
   float max_val = logits[0];
   for(int i = 1; i < ArraySize(logits); i++)
      if(logits[i] > max_val) max_val = logits[i];

   float sum = 0.0;
   for(int i = 0; i < ArraySize(logits); i++)
   {
      proba[i] = (float)MathExp(logits[i] - max_val);
      sum += proba[i];
   }
   for(int i = 0; i < ArraySize(logits); i++)
      proba[i] /= sum;
}

//+------------------------------------------------------------------+
bool HasPosition(ENUM_POSITION_TYPE type)
{
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      PositionGetTicket(i);
      if(PositionGetString(POSITION_SYMBOL) == _Symbol &&
         PositionGetInteger(POSITION_MAGIC)  == MagicNumber &&
         (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE) == type)
         return true;
   }
   return false;
}

void CloseByType(ENUM_POSITION_TYPE type)
{
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong ticket = PositionGetTicket(i);
      if(PositionGetString(POSITION_SYMBOL) == _Symbol &&
         PositionGetInteger(POSITION_MAGIC)  == MagicNumber &&
         (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE) == type)
         g_trade.PositionClose(ticket);
   }
}
//+------------------------------------------------------------------+
