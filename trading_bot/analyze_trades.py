import os
import argparse
from src.analyzer import TradingAnalyzer

def main():
    parser = argparse.ArgumentParser(description='Analyze trading opportunities')
    parser.add_argument('--model', type=str, help='Path to saved model file (.h5)')
    parser.add_argument('--symbol', type=str, help='Symbol to analyze (e.g., BTC/USDT or AAPL)')
    parser.add_argument('--portfolio', action='store_true', help='Analyze entire portfolio')
    parser.add_argument('--list-assets', action='store_true', help='List available assets')
    parser.add_argument('--output', type=str, help='Save analysis to file')
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = TradingAnalyzer(args.model if args.model else None)
    
    # List available assets
    if args.list_assets:
        assets = analyzer.list_available_assets()
        print("\nAvailable Assets:")
        for category, symbols in assets.items():
            print(f"\n{category.upper()}:")
            for symbol in symbols:
                print(f"  - {symbol}")
        return
    
    # Analyze single symbol
    if args.symbol:
        print(f"\nAnalyzing {args.symbol}...")
        recommendation = analyzer.get_recommendation(args.symbol)
        if recommendation:
            print("\nRecommendation Summary:")
            print(f"Symbol: {recommendation['symbol']}")
            print(f"Timestamp: {recommendation['timestamp']}")
            print("\nFinal Recommendation:")
            final_rec = recommendation['final_recommendation']
            print(f"Action: {final_rec['action']}")
            print(f"Confidence: {final_rec['confidence']:.2f}")
            
            # Show detailed scores
            print("\nDetailed Scores:")
            for action, score in final_rec['action_scores'].items():
                print(f"{action}: {score:.2f}")
            
            # Show source-specific recommendations
            if 'recommendations' in recommendation:
                print("\nSource-specific Recommendations:")
                for source, rec in recommendation['recommendations'].items():
                    print(f"\n{source.upper()}:")
                    print(f"Action: {rec['action']}")
                    print(f"Confidence: {rec['confidence']:.2f}")
            
            # Save to file if requested
            if args.output:
                analyzer.save_recommendation(recommendation, args.output)
    
    # Analyze portfolio
    elif args.portfolio:
        print("\nAnalyzing portfolio...")
        # Get all supported assets
        assets = analyzer.list_available_assets()
        all_symbols = []
        for symbols in assets.values():
            all_symbols.extend(symbols)
        
        # Analyze portfolio
        portfolio_analysis = analyzer.analyze_portfolio(all_symbols)
        
        # Print summary
        print("\nPortfolio Analysis Summary:")
        summary = portfolio_analysis['summary']
        print(f"Total Assets Analyzed: {summary['total_assets']}")
        print("\nRecommendations Distribution:")
        for action, count in summary['recommendations'].items():
            print(f"{action}: {count}")
        
        print("\nHigh Confidence Signals:")
        for signal in summary['high_confidence_signals']:
            print(f"{signal['symbol']}: {signal['action']} (confidence: {signal['confidence']:.2f})")
        
        print(f"\nAverage Confidence: {summary['average_confidence']:.2f}")
        
        # Save to file if requested
        if args.output:
            analyzer.save_recommendation(portfolio_analysis, args.output)
    
    else:
        print("Please specify either --symbol or --portfolio, or use --list-assets to see available assets")

if __name__ == "__main__":
    main() 