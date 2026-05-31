//+------------------------------------------------------------------+
//|  MLAdvisor.mq5                                                   |
//|  Expert Advisor — Memanggil Flask API untuk sinyal ML            |
//|                                                                  |
//|  Setup:                                                          |
//|  1. Jalankan python api/server.py terlebih dahulu                |
//|  2. Di MT5: Tools → Options → Expert Advisors                   |
//|     Centang "Allow WebRequest for listed URL"                    |
//|     Tambahkan: http://127.0.0.1:5000                             |
//|  3. Compile dan attach ke chart                                  |
//+------------------------------------------------------------------+
#property copyright "ML Forex Advisor"
#property version   "1.00"
#property strict

#include <Trade\Trade.mqh>
#include <Trade\SymbolInfo.mqh>
#include <Strings\String.mqh>

//--- Input parameters
input string   API_URL           = "http://127.0.0.1:5000";  // URL API Server
input string   InpSymbol         = "";                         // Symbol (kosong = chart symbol)
input string   InpTimeframe      = "M5";                       // Timeframe data
input int      InpBarsToSend     = 300;                        // Jumlah bar dikirim ke API
input double   InpLotSize        = 0.0;                        // Lot manual (0 = auto dari API)
input int      InpMagicNumber    = 20240101;                   // Magic number
input bool     InpEnableTrading  = true;                       // Aktifkan eksekusi order
input int      InpMaxSlippage    = 3;                          // Slippage maks (pips)
input bool     InpCloseOpposite  = true;                       // Tutup posisi berlawanan
input int      InpApiTimeoutMs   = 5000;                       // Timeout request API (ms)
input int      InpUpdateEveryBars = 1;                         // Update setiap N bar baru

//--- Global variables
CTrade         g_trade;
CSymbolInfo    g_symbol;
string         g_symbol_name;
datetime       g_last_bar_time  = 0;
int            g_bars_since_update = 0;
string         g_last_signal    = "HOLD";
double         g_last_confidence = 0.0;
datetime       g_last_request   = 0;

//+------------------------------------------------------------------+
//| Expert initialization                                            |
//+------------------------------------------------------------------+
int OnInit()
{
   g_symbol_name = (StringLen(InpSymbol) > 0) ? InpSymbol : _Symbol;

   if(!g_symbol.Name(g_symbol_name))
   {
      Print("ERROR: Symbol ", g_symbol_name, " tidak ditemukan!");
      return INIT_FAILED;
   }

   g_trade.SetExpertMagicNumber(InpMagicNumber);
   g_trade.SetDeviationInPoints(InpMaxSlippage * 10);
   g_trade.SetTypeFilling(ORDER_FILLING_IOC);

   // Test koneksi ke API
   string response = "";
   int http_code = SendRequest(API_URL + "/health", "", response);
   if(http_code == 200)
      Print("✅ Koneksi ke API server berhasil");
   else
      Print("⚠️  API server tidak merespons (code=", http_code, "). Pastikan server berjalan.");

   Print("MLAdvisor initialized | Symbol: ", g_symbol_name, " | Magic: ", InpMagicNumber);
   return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| Expert deinitialization                                          |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   Print("MLAdvisor dihentikan. Alasan: ", reason);
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
{
   // Cek apakah bar baru terbentuk
   datetime current_bar = iTime(g_symbol_name, PERIOD_CURRENT, 0);
   if(current_bar == g_last_bar_time)
      return;  // Masih di bar yang sama

   g_bars_since_update++;
   g_last_bar_time = current_bar;

   if(g_bars_since_update < InpUpdateEveryBars)
      return;

   g_bars_since_update = 0;
   g_last_request = TimeCurrent();

   // Ambil sinyal dari API
   string signal_json = GetSignalFromAPI();
   if(StringLen(signal_json) == 0)
   {
      Print("Tidak ada respons dari API.");
      return;
   }

   // Parse JSON response
   string signal = "";
   double confidence = 0, sl_price = 0, tp_price = 0, lot_size = 0;
   bool risk_passed = false;
   ParseSignalJSON(signal_json, signal, confidence, sl_price, tp_price, lot_size, risk_passed);

   // Display info di chart
   DisplayInfo(signal, confidence, sl_price, tp_price, lot_size, risk_passed);

   // Eksekusi order jika memenuhi syarat
   if(!InpEnableTrading)
   {
      Print("Trading disabled. Signal: ", signal, " | Conf: ", DoubleToString(confidence, 3));
      return;
   }

   if(!risk_passed)
   {
      Print("Risk filter tidak lolos. Signal diabaikan.");
      return;
   }

   if(signal == "BUY")
      ExecuteBuy(lot_size, sl_price, tp_price);
   else if(signal == "SELL")
      ExecuteSell(lot_size, sl_price, tp_price);
   // HOLD: tidak ada aksi

   g_last_signal = signal;
   g_last_confidence = confidence;
}

//+------------------------------------------------------------------+
//| Kirim data ke API dan terima prediksi                           |
//+------------------------------------------------------------------+
string GetSignalFromAPI()
{
   // Build OHLCV JSON payload
   MqlRates rates[];
   ENUM_TIMEFRAMES tf = StringToTimeframe(InpTimeframe);
   int copied = CopyRates(g_symbol_name, tf, 0, InpBarsToSend, rates);
   if(copied <= 0)
   {
      Print("Gagal copy rates: ", GetLastError());
      return "";
   }

   // Bangun JSON array bar
   string bars_json = "";
   for(int i = 0; i < copied; i++)
   {
      string bar = StringFormat(
         "{\"time\":\"%s\",\"open\":%.5f,\"high\":%.5f,\"low\":%.5f,\"close\":%.5f,\"volume\":%I64d}",
         TimeToString(rates[i].time, TIME_DATE|TIME_MINUTES),
         rates[i].open, rates[i].high, rates[i].low, rates[i].close,
         rates[i].tick_volume
      );
      bars_json += bar;
      if(i < copied - 1) bars_json += ",";
   }

   string body = StringFormat(
      "{\"symbol\":\"%s\",\"timeframe\":\"%s\",\"bars\":[%s]}",
      g_symbol_name, InpTimeframe, bars_json
   );

   string response = "";
   int code = SendRequest(API_URL + "/predict", body, response);

   if(code != 200)
   {
      Print("API error. HTTP code: ", code);
      return "";
   }
   return response;
}

//+------------------------------------------------------------------+
//| HTTP POST/GET helper                                             |
//+------------------------------------------------------------------+
int SendRequest(string url, string body, string &response)
{
   char   post_data[];
   char   result[];
   string result_headers;
   string headers = "Content-Type: application/json\r\n";

   if(StringLen(body) > 0)
      StringToCharArray(body, post_data, 0, StringLen(body));
   else
      ArrayResize(post_data, 0);

   int res = WebRequest(
      StringLen(body) > 0 ? "POST" : "GET",
      url,
      headers,
      InpApiTimeoutMs,
      post_data,
      result,
      result_headers
   );

   if(res == -1)
   {
      int err = GetLastError();
      if(err == 4014)
         Print("WebRequest tidak diizinkan. Tambahkan URL ke 'Allow WebRequest' di Options.");
      else
         Print("WebRequest error: ", err);
      return -1;
   }

   response = CharArrayToString(result, 0, WHOLE_ARRAY, CP_UTF8);
   return res;
}

//+------------------------------------------------------------------+
//| Parse JSON response (simple parser tanpa library eksternal)      |
//+------------------------------------------------------------------+
void ParseSignalJSON(
   const string json,
   string &signal,
   double &confidence,
   double &sl_price,
   double &tp_price,
   double &lot_size,
   bool &risk_passed
)
{
   signal      = ExtractString(json, "signal");
   confidence  = ExtractDouble(json, "confidence");
   sl_price    = ExtractDouble(json, "sl_price");
   tp_price    = ExtractDouble(json, "tp_price");
   lot_size    = ExtractDouble(json, "lot_size");
   string rp   = ExtractString(json, "risk_passed");
   risk_passed = (rp == "true");

   Print(StringFormat("Signal: %s | Conf: %.3f | SL: %.5f | TP: %.5f | Lot: %.2f | Risk OK: %s",
         signal, confidence, sl_price, tp_price, lot_size, risk_passed ? "Ya" : "Tidak"));
}

string ExtractString(const string json, const string key)
{
   string search = "\"" + key + "\":\"";
   int start = StringFind(json, search);
   if(start < 0) return "";
   start += StringLen(search);
   int end = StringFind(json, "\"", start);
   if(end < 0) return "";
   return StringSubstr(json, start, end - start);
}

double ExtractDouble(const string json, const string key)
{
   string search = "\"" + key + "\":";
   int start = StringFind(json, search);
   if(start < 0) return 0.0;
   start += StringLen(search);
   // Baca hingga delimiter berikutnya
   string val = "";
   for(int i = start; i < MathMin(start + 20, StringLen(json)); i++)
   {
      string ch = StringSubstr(json, i, 1);
      if(ch == "," || ch == "}" || ch == "]") break;
      val += ch;
   }
   return StringToDouble(val);
}

//+------------------------------------------------------------------+
//| Eksekusi BUY                                                     |
//+------------------------------------------------------------------+
void ExecuteBuy(double lot, double sl, double tp)
{
   // Tutup posisi SELL jika ada
   if(InpCloseOpposite) ClosePositions(POSITION_TYPE_SELL);

   // Skip jika sudah ada posisi BUY
   if(HasPosition(POSITION_TYPE_BUY)) return;

   double volume = (lot > 0 && InpLotSize == 0) ? NormalizeVolume(lot) : NormalizeVolume(InpLotSize > 0 ? InpLotSize : 0.01);
   double ask    = g_symbol.Ask();

   if(!g_trade.Buy(volume, g_symbol_name, ask, sl, tp, "ML-BUY"))
      Print("BUY gagal: ", g_trade.ResultRetcodeDescription());
   else
      Print(StringFormat("✅ BUY %.2f %s @ %.5f | SL:%.5f TP:%.5f", volume, g_symbol_name, ask, sl, tp));
}

//+------------------------------------------------------------------+
//| Eksekusi SELL                                                    |
//+------------------------------------------------------------------+
void ExecuteSell(double lot, double sl, double tp)
{
   if(InpCloseOpposite) ClosePositions(POSITION_TYPE_BUY);
   if(HasPosition(POSITION_TYPE_SELL)) return;

   double volume = (lot > 0 && InpLotSize == 0) ? NormalizeVolume(lot) : NormalizeVolume(InpLotSize > 0 ? InpLotSize : 0.01);
   double bid    = g_symbol.Bid();

   if(!g_trade.Sell(volume, g_symbol_name, bid, sl, tp, "ML-SELL"))
      Print("SELL gagal: ", g_trade.ResultRetcodeDescription());
   else
      Print(StringFormat("✅ SELL %.2f %s @ %.5f | SL:%.5f TP:%.5f", volume, g_symbol_name, bid, sl, tp));
}

//+------------------------------------------------------------------+
//| Cek apakah ada posisi terbuka dengan type tertentu              |
//+------------------------------------------------------------------+
bool HasPosition(ENUM_POSITION_TYPE type)
{
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong ticket = PositionGetTicket(i);
      if(PositionGetString(POSITION_SYMBOL) == g_symbol_name &&
         PositionGetInteger(POSITION_MAGIC) == InpMagicNumber &&
         (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE) == type)
         return true;
   }
   return false;
}

//+------------------------------------------------------------------+
//| Tutup semua posisi dengan type tertentu                         |
//+------------------------------------------------------------------+
void ClosePositions(ENUM_POSITION_TYPE type)
{
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong ticket = PositionGetTicket(i);
      if(PositionGetString(POSITION_SYMBOL) == g_symbol_name &&
         PositionGetInteger(POSITION_MAGIC) == InpMagicNumber &&
         (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE) == type)
      {
         if(!g_trade.PositionClose(ticket))
            Print("Gagal menutup posisi: ", g_trade.ResultRetcodeDescription());
      }
   }
}

//+------------------------------------------------------------------+
//| Normalisasi lot sesuai aturan broker                            |
//+------------------------------------------------------------------+
double NormalizeVolume(double lot)
{
   double min_vol  = g_symbol.LotsMin();
   double max_vol  = g_symbol.LotsMax();
   double step_vol = g_symbol.LotsStep();
   lot = MathMax(min_vol, MathMin(lot, max_vol));
   lot = MathRound(lot / step_vol) * step_vol;
   return NormalizeDouble(lot, 2);
}

//+------------------------------------------------------------------+
//| Konversi string timeframe ke ENUM                               |
//+------------------------------------------------------------------+
ENUM_TIMEFRAMES StringToTimeframe(string tf)
{
   if(tf == "M1")  return PERIOD_M1;
   if(tf == "M5")  return PERIOD_M5;
   if(tf == "M15") return PERIOD_M15;
   if(tf == "M30") return PERIOD_M30;
   if(tf == "H1")  return PERIOD_H1;
   if(tf == "H4")  return PERIOD_H4;
   if(tf == "D1")  return PERIOD_D1;
   return PERIOD_H1;
}

//+------------------------------------------------------------------+
//| Tampilkan info di chart                                         |
//+------------------------------------------------------------------+
void DisplayInfo(string signal, double conf, double sl, double tp, double lot, bool risk_ok)
{
   string color_txt = (signal == "BUY") ? "🟢 BUY" : (signal == "SELL") ? "🔴 SELL" : "⚪ HOLD";
   string info = StringFormat(
      "ML Advisor | %s | Confidence: %.1f%% | Lot: %.2f | Risk: %s\nSL: %.5f | TP: %.5f | Updated: %s",
      color_txt, conf * 100, lot, risk_ok ? "✅" : "❌",
      sl, tp, TimeToString(TimeCurrent(), TIME_DATE|TIME_MINUTES|TIME_SECONDS)
   );
   Comment(info);
}
//+------------------------------------------------------------------+
