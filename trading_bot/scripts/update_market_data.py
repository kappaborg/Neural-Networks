#!/usr/bin/env python3
"""
Market Data Auto Updater
Otomatik piyasa verisi güncelleyici

Bu script'i cron job olarak çalıştırarak verileri otomatik güncelleyebilirsiniz:
# Her gün sabah 9:00'da çalıştırmak için:
# 0 9 * * * /path/to/python /path/to/update_market_data.py

# Her 4 saatte bir çalıştırmak için:
# 0 */4 * * * /path/to/python /path/to/update_market_data.py
"""

import os
import sys
import logging
import schedule
import time
from datetime import datetime
from pathlib import Path

# Proje root'unu Python path'ine ekle
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

from data_updater import DataUpdater

# Logging ayarları
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(project_root / 'logs' / 'data_updater.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class AutoDataUpdater:
    """Otomatik veri güncelleyici"""
    
    def __init__(self):
        self.updater = DataUpdater()
        self.last_update = None
        
        # Log klasörünü oluştur
        log_dir = project_root / 'logs'
        log_dir.mkdir(exist_ok=True)
        
    def update_data(self):
        """Verileri güncelle"""
        try:
            logger.info("🚀 Otomatik veri güncelleme başlatılıyor...")
            
            # Güncellemeden önce mevcut dosyaları listele
            file_count_before = self.updater.list_cache_files()
            logger.info(f"📁 Güncelleme öncesi dosya sayısı: {file_count_before}")
            
            # Verileri güncelle
            successful, total = self.updater.update_all_data()
            
            # Sonuçları logla
            success_rate = (successful / total) * 100 if total > 0 else 0
            logger.info(f"✅ Güncelleme tamamlandı: {successful}/{total} (%{success_rate:.1f})")
            
            # Güncelleme zamanını kaydet
            self.last_update = datetime.now()
            
            # Eski dosyaları temizle (30 günden eski)
            deleted_count = self.updater.clean_old_files(days_old=30)
            if deleted_count > 0:
                logger.info(f"🗑️ {deleted_count} eski dosya temizlendi")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Veri güncelleme hatası: {e}")
            return False
    
    def run_scheduled_updates(self):
        """Zamanlanmış güncellemeleri çalıştır"""
        logger.info("📅 Zamanlanmış güncelleme sistemi başlatılıyor...")
        
        # Her gün sabah 9:00'da güncelle
        schedule.every().day.at("09:00").do(self.update_data)
        
        # Her 6 saatte bir güncelle
        schedule.every(6).hours.do(self.update_data)
        
        # İlk güncellemeyi hemen yap
        self.update_data()
        
        logger.info("⏰ Zamanlanmış görevler ayarlandı:")
        logger.info("  - Her gün 09:00'da")
        logger.info("  - Her 6 saatte bir")
        
        # Ana döngü
        while True:
            try:
                schedule.run_pending()
                time.sleep(60)  # Her dakika kontrol et
                
            except KeyboardInterrupt:
                logger.info("👋 Güncelleme sistemi kapatılıyor...")
                break
            except Exception as e:
                logger.error(f"❌ Zamanlanmış güncelleme hatası: {e}")
                time.sleep(300)  # 5 dakika bekle ve devam et
    
    def run_single_update(self):
        """Tek seferlik güncelleme yap"""
        logger.info("🔄 Tek seferlik güncelleme yapılıyor...")
        success = self.update_data()
        
        if success:
            logger.info("✅ Güncelleme başarıyla tamamlandı!")
        else:
            logger.error("❌ Güncelleme başarısız!")
        
        return success

def main():
    """Ana fonksiyon"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Market Data Auto Updater')
    parser.add_argument('--mode', choices=['once', 'scheduled'], default='once',
                       help='Güncelleme modu: once (tek seferlik) veya scheduled (zamanlanmış)')
    parser.add_argument('--verbose', action='store_true',
                       help='Detaylı log çıktısı')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Auto updater'ı başlat
    auto_updater = AutoDataUpdater()
    
    if args.mode == 'once':
        # Tek seferlik güncelleme
        success = auto_updater.run_single_update()
        sys.exit(0 if success else 1)
    
    elif args.mode == 'scheduled':
        # Zamanlanmış güncellemeler
        auto_updater.run_scheduled_updates()

if __name__ == "__main__":
    main() 