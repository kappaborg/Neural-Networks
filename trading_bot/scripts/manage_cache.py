#!/usr/bin/env python3
"""
Cache Management Tool
Cache yönetim aracı

Bu araç ile cache dosyalarını yönetebilir, optimize edebilir ve analiz edebilirsiniz.
"""

import os
import sys
import json
import argparse
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd

# Proje root'unu Python path'ine ekle
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

from data_updater import DataUpdater

class CacheManager:
    """Cache yönetim sınıfı"""
    
    def __init__(self, cache_dir="data/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def analyze_cache(self):
        """Cache analizi yap"""
        print("🔍 CACHE ANALİZİ")
        print("="*50)
        
        files = list(self.cache_dir.glob("*.csv"))
        
        if not files:
            print("❌ Cache'de dosya bulunamadı!")
            return
        
        total_size = 0
        crypto_files = []
        stock_files = []
        file_info = []
        
        for file in files:
            file_size = file.stat().st_size
            total_size += file_size
            
            # Dosya detaylarını çıkar
            name_parts = file.stem.split('_')
            if len(name_parts) >= 3:
                symbol = name_parts[0] + ('_' + name_parts[1] if 'USDT' in name_parts[1] else '')
                start_date = name_parts[-2] if len(name_parts) >= 3 else 'Unknown'
                end_date = name_parts[-1] if len(name_parts) >= 3 else 'Unknown'
            else:
                symbol = file.stem
                start_date = 'Unknown'
                end_date = 'Unknown'
            
            # Dosya içeriğini analiz et
            try:
                df = pd.read_csv(file)
                row_count = len(df)
                columns = list(df.columns)
                
                file_data = {
                    'file': file.name,
                    'symbol': symbol,
                    'size_kb': file_size / 1024,
                    'rows': row_count,
                    'columns': len(columns),
                    'start_date': start_date,
                    'end_date': end_date,
                    'modified': datetime.fromtimestamp(file.stat().st_mtime)
                }
                
                file_info.append(file_data)
                
                if 'USDT' in symbol:
                    crypto_files.append(file_data)
                else:
                    stock_files.append(file_data)
                    
            except Exception as e:
                print(f"⚠️ {file.name} dosyası okunamadı: {e}")
        
        # Genel istatistikler
        print(f"📁 Toplam dosya sayısı: {len(files)}")
        print(f"💾 Toplam boyut: {total_size / (1024*1024):.2f} MB")
        print(f"🪙 Kripto dosyaları: {len(crypto_files)}")
        print(f"📈 Hisse senedi dosyaları: {len(stock_files)}")
        
        # En büyük ve en küçük dosyalar
        if file_info:
            largest = max(file_info, key=lambda x: x['size_kb'])
            smallest = min(file_info, key=lambda x: x['size_kb'])
            
            print(f"\n📏 En büyük dosya: {largest['file']} ({largest['size_kb']:.1f} KB)")
            print(f"📏 En küçük dosya: {smallest['file']} ({smallest['size_kb']:.1f} KB)")
            
            # En güncel ve en eski dosyalar
            newest = max(file_info, key=lambda x: x['modified'])
            oldest = min(file_info, key=lambda x: x['modified'])
            
            print(f"🕐 En güncel: {newest['file']} ({newest['modified'].strftime('%Y-%m-%d %H:%M')})")
            print(f"🕘 En eski: {oldest['file']} ({oldest['modified'].strftime('%Y-%m-%d %H:%M')})")
        
        return file_info
    
    def clean_duplicates(self):
        """Duplicate dosyaları temizle"""
        print("\n🧹 DUPLICATE DOSYA TEMİZLİĞİ")
        print("="*50)
        
        files = list(self.cache_dir.glob("*.csv"))
        symbol_files = {}
        
        # Sembollere göre grupla
        for file in files:
            name_parts = file.stem.split('_')
            if len(name_parts) >= 2:
                symbol = name_parts[0] + ('_' + name_parts[1] if 'USDT' in name_parts[1] else '')
            else:
                symbol = name_parts[0]
            
            if symbol not in symbol_files:
                symbol_files[symbol] = []
            
            symbol_files[symbol].append(file)
        
        deleted_count = 0
        
        # Her sembol için en güncel dosyayı tut, eskilerini sil
        for symbol, files_list in symbol_files.items():
            if len(files_list) > 1:
                # Dosyaları değiştirilme tarihine göre sırala
                files_list.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                
                print(f"\n🔄 {symbol} için {len(files_list)} dosya bulundu:")
                
                # En güncel dosyayı koru
                keep_file = files_list[0]
                print(f"✅ Korunacak: {keep_file.name}")
                
                # Eski dosyaları sil
                for old_file in files_list[1:]:
                    print(f"🗑️ Siliniyor: {old_file.name}")
                    old_file.unlink()
                    deleted_count += 1
        
        print(f"\n✅ {deleted_count} duplicate dosya temizlendi")
        return deleted_count
    
    def clean_old_files(self, days_old=30):
        """Eski dosyaları temizle"""
        print(f"\n🗑️ ESKİ DOSYA TEMİZLİĞİ ({days_old} günden eski)")
        print("="*50)
        
        cutoff_date = datetime.now() - timedelta(days=days_old)
        deleted_count = 0
        
        for file in self.cache_dir.glob("*.csv"):
            file_date = datetime.fromtimestamp(file.stat().st_mtime)
            
            if file_date < cutoff_date:
                print(f"🗑️ Siliniyor: {file.name} (Tarih: {file_date.strftime('%Y-%m-%d')})")
                file.unlink()
                deleted_count += 1
        
        print(f"✅ {deleted_count} eski dosya temizlendi")
        return deleted_count
    
    def export_cache_info(self, output_file="cache_info.json"):
        """Cache bilgilerini JSON olarak dışa aktar"""
        print(f"\n📤 CACHE BİLGİLERİ EXPORT EDİLİYOR: {output_file}")
        print("="*50)
        
        file_info = []
        
        for file in self.cache_dir.glob("*.csv"):
            try:
                df = pd.read_csv(file)
                
                info = {
                    'filename': file.name,
                    'path': str(file),
                    'size_bytes': file.stat().st_size,
                    'size_kb': file.stat().st_size / 1024,
                    'rows': len(df),
                    'columns': list(df.columns),
                    'column_count': len(df.columns),
                    'created': datetime.fromtimestamp(file.stat().st_ctime).isoformat(),
                    'modified': datetime.fromtimestamp(file.stat().st_mtime).isoformat(),
                }
                
                # Veri aralığını çıkar
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    info['date_range'] = {
                        'start': df['timestamp'].min().isoformat(),
                        'end': df['timestamp'].max().isoformat()
                    }
                
                # OHLCV verilerini çıkar
                if all(col in df.columns for col in ['Open', 'High', 'Low', 'Close']):
                    info['price_range'] = {
                        'min': float(df['Low'].min()),
                        'max': float(df['High'].max()),
                        'last': float(df['Close'].iloc[-1])
                    }
                
                file_info.append(info)
                
            except Exception as e:
                print(f"⚠️ {file.name} işlenemedi: {e}")
        
        # JSON olarak kaydet
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'export_date': datetime.now().isoformat(),
                'total_files': len(file_info),
                'files': file_info
            }, f, indent=2, ensure_ascii=False)
        
        print(f"✅ {len(file_info)} dosya bilgisi {output_file} dosyasına kaydedildi")
        return len(file_info)
    
    def optimize_cache(self):
        """Cache'i optimize et"""
        print("\n⚡ CACHE OPTİMİZASYONU")
        print("="*50)
        
        operations = []
        
        # Duplicate temizliği
        duplicates_cleaned = self.clean_duplicates()
        operations.append(f"Duplicate temizlik: {duplicates_cleaned} dosya")
        
        # Eski dosya temizliği
        old_files_cleaned = self.clean_old_files(days_old=60)
        operations.append(f"Eski dosya temizlik: {old_files_cleaned} dosya")
        
        # Cache bilgilerini export et
        exported_count = self.export_cache_info()
        operations.append(f"Bilgi export: {exported_count} dosya")
        
        print(f"\n✅ CACHE OPTİMİZASYONU TAMAMLANDI")
        for op in operations:
            print(f"  - {op}")
        
        return operations

def main():
    """Ana fonksiyon"""
    parser = argparse.ArgumentParser(description='Cache Management Tool')
    parser.add_argument('action', choices=['analyze', 'clean-duplicates', 'clean-old', 'export', 'optimize'],
                       help='Yapılacak işlem')
    parser.add_argument('--days', type=int, default=30,
                       help='Eski dosya temizliği için gün sayısı (varsayılan: 30)')
    parser.add_argument('--output', default='cache_info.json',
                       help='Export dosyası adı (varsayılan: cache_info.json)')
    
    args = parser.parse_args()
    
    cache_manager = CacheManager()
    
    if args.action == 'analyze':
        cache_manager.analyze_cache()
        
    elif args.action == 'clean-duplicates':
        cache_manager.clean_duplicates()
        
    elif args.action == 'clean-old':
        cache_manager.clean_old_files(args.days)
        
    elif args.action == 'export':
        cache_manager.export_cache_info(args.output)
        
    elif args.action == 'optimize':
        cache_manager.optimize_cache()

if __name__ == "__main__":
    main() 