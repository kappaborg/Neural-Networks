import yfinance as yf
import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import time
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class DataUpdater:
    """
    Veri dosyalarını güncelleyen ve genişleten sistem
    """
    
    def __init__(self, cache_dir="data/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Kripto verileri için Binance exchange
        self.exchange = ccxt.binance({
            'apiKey': '',  # İsteğe bağlı
            'secret': '',  # İsteğe bağlı
            'timeout': 30000,
            'enableRateLimit': True
        })
        
        # Desteklenen semboller
        self.crypto_symbols = [
            'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'ADA/USDT', 'SOL/USDT',
            'XRP/USDT', 'DOT/USDT', 'DOGE/USDT', 'AVAX/USDT', 'MATIC/USDT',
            'LINK/USDT', 'UNI/USDT', 'LTC/USDT', 'TRX/USDT', 'ATOM/USDT'
        ]
        
        self.stock_symbols = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META',
            'NFLX', 'ADBE', 'CRM', 'ORCL', 'IBM', 'INTC', 'AMD', 'UBER',
            'JPM', 'BAC', 'WFC', 'GS', 'MS', 'V', 'MA', 'PYPL'
        ]
        
        self.crypto_fear_greed_index = 33  # Default value
        
    def calculate_technical_indicators(self, df):
        """Teknik göstergeleri hesapla"""
        if len(df) < 200:
            return df
            
        try:
            # Moving Averages
            df['SMA_20'] = df['Close'].rolling(window=20).mean()
            df['SMA_50'] = df['Close'].rolling(window=50).mean()
            df['SMA_200'] = df['Close'].rolling(window=200).mean()
            
            # Exponential Moving Averages
            df['EMA_12'] = df['Close'].ewm(span=12).mean()
            df['EMA_26'] = df['Close'].ewm(span=26).mean()
            
            # MACD
            df['MACD'] = df['EMA_12'] - df['EMA_26']
            df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
            
            # RSI
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # Stochastic RSI
            rsi_min = df['RSI'].rolling(window=14).min()
            rsi_max = df['RSI'].rolling(window=14).max()
            df['Stoch_RSI'] = (df['RSI'] - rsi_min) / (rsi_max - rsi_min)
            
            # Bollinger Bands
            bb_period = 20
            bb_std = 2
            df['BB_Middle'] = df['Close'].rolling(window=bb_period).mean()
            bb_std_dev = df['Close'].rolling(window=bb_period).std()
            df['BB_Upper'] = df['BB_Middle'] + (bb_std_dev * bb_std)
            df['BB_Lower'] = df['BB_Middle'] - (bb_std_dev * bb_std)
            
            # Average True Range (ATR)
            high_low = df['High'] - df['Low']
            high_close = np.abs(df['High'] - df['Close'].shift())
            low_close = np.abs(df['Low'] - df['Close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            df['ATR'] = true_range.rolling(window=14).mean()
            
            # On Balance Volume (OBV)
            df['OBV'] = 0.0
            for i in range(1, len(df)):
                if df['Close'].iloc[i] > df['Close'].iloc[i-1]:
                    df['OBV'].iloc[i] = df['OBV'].iloc[i-1] + df['Volume'].iloc[i]
                elif df['Close'].iloc[i] < df['Close'].iloc[i-1]:
                    df['OBV'].iloc[i] = df['OBV'].iloc[i-1] - df['Volume'].iloc[i]
                else:
                    df['OBV'].iloc[i] = df['OBV'].iloc[i-1]
            
            # Accumulation/Distribution Line (ADI)
            clv = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'])
            clv = clv.fillna(0.0)
            df['ADI'] = (clv * df['Volume']).cumsum()
            
            # Fear & Greed Index (sabit değer)
            df['Fear_Greed_Index'] = self.crypto_fear_greed_index
            
        except Exception as e:
            print(f"Teknik gösterge hesaplama hatası: {e}")
            
        return df
    
    def fetch_crypto_data(self, symbol, timeframe='1d', limit=500):
        """Kripto verilerini çek"""
        try:
            # Son limit günlük veri çek
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            
            if not ohlcv:
                return None
                
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Teknik göstergeleri hesapla
            df = self.calculate_technical_indicators(df)
            
            return df
            
        except Exception as e:
            print(f"Kripto veri çekme hatası {symbol}: {e}")
            return None
    
    def fetch_stock_data(self, symbol, period='2y'):
        """Hisse senedi verilerini çek"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            
            if data.empty:
                return None
                
            # Reset index to get Date as column
            df = data.reset_index()
            df.rename(columns={'Date': 'timestamp'}, inplace=True)
            
            # Teknik göstergeleri hesapla
            df = self.calculate_technical_indicators(df)
            
            return df
            
        except Exception as e:
            print(f"Hisse senedi veri çekme hatası {symbol}: {e}")
            return None
    
    def save_data(self, df, symbol, data_type):
        """Veriyi dosyaya kaydet"""
        if df is None or df.empty:
            return False
            
        try:
            # Dosya adını oluştur
            clean_symbol = symbol.replace('/', '_')
            start_date = df['timestamp'].min().strftime('%Y-%m-%d')
            end_date = df['timestamp'].max().strftime('%Y-%m-%d')
            filename = f"{clean_symbol}_{start_date}_{end_date}.csv"
            filepath = self.cache_dir / filename
            
            # Veriyi kaydet
            df.to_csv(filepath, index=False)
            print(f"✅ {symbol} verisi kaydedildi: {filename}")
            return True
            
        except Exception as e:
            print(f"❌ Veri kaydetme hatası {symbol}: {e}")
            return False
    
    def update_crypto_data(self):
        """Tüm kripto verilerini güncelle"""
        print("🔄 Kripto verileri güncelleniyor...")
        
        successful = 0
        total = len(self.crypto_symbols)
        
        for symbol in self.crypto_symbols:
            print(f"📊 İşleniyor: {symbol}")
            
            # Veriyi çek
            df = self.fetch_crypto_data(symbol)
            
            # Kaydet
            if self.save_data(df, symbol, 'crypto'):
                successful += 1
            
            # Rate limit için bekle
            time.sleep(0.5)
        
        print(f"✅ Kripto veri güncellemesi tamamlandı: {successful}/{total}")
        return successful, total
    
    def update_stock_data(self):
        """Tüm hisse senedi verilerini güncelle"""
        print("🔄 Hisse senedi verileri güncelleniyor...")
        
        successful = 0
        total = len(self.stock_symbols)
        
        for symbol in self.stock_symbols:
            print(f"📊 İşleniyor: {symbol}")
            
            # Veriyi çek
            df = self.fetch_stock_data(symbol)
            
            # Kaydet
            if self.save_data(df, symbol, 'stock'):
                successful += 1
            
            # Rate limit için bekle
            time.sleep(0.2)
        
        print(f"✅ Hisse senedi veri güncellemesi tamamlandı: {successful}/{total}")
        return successful, total
    
    def update_all_data(self):
        """Tüm verileri güncelle"""
        print("🚀 Veri güncelleme işlemi başlatılıyor...")
        start_time = time.time()
        
        # Kripto verilerini güncelle
        crypto_success, crypto_total = self.update_crypto_data()
        
        # Hisse senedi verilerini güncelle
        stock_success, stock_total = self.update_stock_data()
        
        # Özet
        total_time = time.time() - start_time
        total_success = crypto_success + stock_success
        total_symbols = crypto_total + stock_total
        
        print("\n" + "="*50)
        print("📊 VERİ GÜNCELLEME ÖZETİ")
        print("="*50)
        print(f"Kripto semboller: {crypto_success}/{crypto_total}")
        print(f"Hisse senetleri: {stock_success}/{stock_total}")
        print(f"Toplam başarılı: {total_success}/{total_symbols}")
        print(f"Başarı oranı: {(total_success/total_symbols)*100:.1f}%")
        print(f"Süre: {total_time:.1f} saniye")
        print("="*50)
        
        return total_success, total_symbols
    
    def list_cache_files(self):
        """Cache dosyalarını listele"""
        files = list(self.cache_dir.glob("*.csv"))
        
        print(f"\n📁 Cache klasöründe {len(files)} dosya bulundu:")
        print("-"*60)
        
        crypto_files = []
        stock_files = []
        
        for file in files:
            file_size = file.stat().st_size / 1024  # KB
            if '_USDT_' in file.name:
                crypto_files.append((file.name, file_size))
            else:
                stock_files.append((file.name, file_size))
        
        if crypto_files:
            print("🪙 Kripto Verileri:")
            for name, size in sorted(crypto_files):
                print(f"  {name:<40} ({size:>6.1f} KB)")
        
        if stock_files:
            print("\n📈 Hisse Senedi Verileri:")
            for name, size in sorted(stock_files):
                print(f"  {name:<40} ({size:>6.1f} KB)")
        
        print("-"*60)
        return len(files)
    
    def clean_old_files(self, days_old=30):
        """Eski dosyaları temizle"""
        cutoff_date = datetime.now() - timedelta(days=days_old)
        deleted_count = 0
        
        for file in self.cache_dir.glob("*.csv"):
            if file.stat().st_mtime < cutoff_date.timestamp():
                file.unlink()
                print(f"🗑️ Eski dosya silindi: {file.name}")
                deleted_count += 1
        
        print(f"✅ {deleted_count} eski dosya temizlendi")
        return deleted_count

if __name__ == "__main__":
    # Data updater'ı başlat
    updater = DataUpdater()
    
    # Mevcut dosyaları listele
    updater.list_cache_files()
    
    # Tüm verileri güncelle
    updater.update_all_data()
    
    # Güncellenmiş dosyaları listele
    print("\n📊 Güncelleme sonrası:")
    updater.list_cache_files() 