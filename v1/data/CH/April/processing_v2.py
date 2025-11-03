"""
OPTIMIZED COMPANIES HOUSE PARSER - MEMORY LIGHT + ALL STRATEGIES
Combines:
- All 12 original parsing strategies for maximum data extraction
- Pre-compiled regex patterns for speed
- Parallel processing
- Memory-light mode (doesn't hold all years in RAM)
"""

import zipfile
import xml.etree.ElementTree as ET
import pandas as pd
import pickle
import os
import random
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import re
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

# ==========================================================================
# CONFIGURATION
# ==========================================================================

TEST_MODE = False  # Set to True for testing with a sample of files
SAMPLE_SIZE = 100
SAVE_FAILED_FILES = True
MEMORY_LIGHT_MODE = True  # ✅ Memory efficient - enabled by default
NUM_WORKERS = 4 # Parallel processing threads

# ==========================================================================
# METRICS
# ==========================================================================

METRICS = {
    'tangible_assets': ['TangibleFixedAssets', 'TangibleAssets'],
    'intangible_assets': ['IntangibleFixedAssets', 'IntangibleAssets'],
    'fixed_assets': ['FixedAssets', 'NonCurrentAssets'],
    'stocks': ['StocksInventory', 'Stocks', 'Inventories'],
    'debtors': ['Debtors', 'TradeOtherReceivables'],
    'cash': ['CashBankInHand', 'CashBankOnHand', 'CashCashEquivalents'],
    'current_assets': ['CurrentAssets'],
    'creditors_1yr': ['CreditorsDueWithinOneYearTotalCurrentLiabilities', 
                      'CreditorsAmountsFallingDueWithinOneYear', 'CurrentLiabilities'],
    'creditors_after_1yr': ['CreditorsDueAfterOneYearTotalNoncurrentLiabilities',
                            'CreditorsAmountsFallingDueAfterMoreThanOneYear', 'Creditors'],
    'provisions': ['ProvisionsForLiabilitiesCharges', 'Provisions',
                   'ProvisionsForLiabilitiesBalanceSheetSubtotal'],
    'net_current_assets': ['NetCurrentAssetsLiabilities', 'NetCurrentAssets'],
    'total_assets_less_current_liab': ['TotalAssetsLessCurrentLiabilities'],
    'net_assets': ['NetAssetsLiabilitiesIncludingPensionAssetLiability', 
                   'NetAssetsLiabilities', 'NetAssets'],
    'share_capital': ['CalledUpShareCapital', 'ShareCapital'],
    'equity': ['ShareholderFunds', 'Equity', 'TotalEquity'],
    'turnover': ['Turnover', 'TurnoverRevenue', 'Revenue'],
}

# ==========================================================================
# COMPILED REGEX PATTERNS (HUGE SPEEDUP)
# ==========================================================================

COMPILED_PATTERNS = {
    'company_number_filename': re.compile(r'_(OC\d{6}|\d{8})_'),
    'company_number_xml': [
        re.compile(r'(?:identifier|RegisteredNumber|UKCompaniesHouseRegisteredNumber)[^>]*>(OC\d{6}|0*\d{5,8})<', re.IGNORECASE),
        re.compile(r'scheme="http://www\.companieshouse\.gov\.uk/">(OC\d{6}|\d{5,8})</identifier>'),
        re.compile(r'Registration\s+(?:Number|No\.?):?\s*(OC\d{6}|\d{8})', re.IGNORECASE),
    ],
    'period_end_filename': re.compile(r'_(\d{8})\.'),
    'period_end_xml': [
        re.compile(r'(?:EndDateForPeriodCoveredByReport|BalanceSheetDate)[^>]*>(\d{4}-\d{2}-\d{2})<'),
        re.compile(r'<endDate>(\d{4}-\d{2}-\d{2})</endDate>'),
        re.compile(r'<instant>(\d{4}-\d{2}-\d{2})</instant>'),
    ],
    'context': re.compile(r'contextRef="([^"]*(?:CurrYear|FY|cfwd)[^"]*)"'),
    'financial': {
        'current_assets': re.compile(r'CurrentAssets[^>]*>([0-9,]+)'),
        'net_assets': re.compile(r'NetAssets[^>]*>([0-9,]+)'),
        'equity': re.compile(r'(?:Equity|ShareholderFunds)[^>]*>([0-9,]+)'),
        'turnover': re.compile(r'(?:Turnover|Revenue)[^>]*>([0-9,]+)'),
    }
}

# Build lowercase pattern lookup
TAG_TO_METRIC = {}
for metric, patterns in METRICS.items():
    for pattern in patterns:
        TAG_TO_METRIC[pattern.lower()] = metric

# ==========================================================================
# UTILITIES
# ==========================================================================

def clean_number(text):
    """Extract number from text, returns 0.0 for dash/nil, None for invalid"""
    if text is None:
        return None
    try:
        text = str(text).strip()
        if text in ['-', '–', '—', 'nil', 'Nil', 'NIL', '']:
            return 0.0
        text = re.sub(r'[£$€,\s]', '', text)
        text = re.sub(r'[^\d\.\-]', '', text)
        if not text or text == '-':
            return 0.0
        return float(text)
    except:
        return None

# ==========================================================================
# ALL 12 PARSING STRATEGIES (OPTIMIZED WITH COMPILED PATTERNS)
# ==========================================================================

def strategy_1_ixbrl(root, data):
    """iXBRL HTML (2013+ inline XBRL)"""
    ns = {'ix': 'http://www.xbrl.org/2013/inlineXBRL'}
    try:
        for elem in root.findall('.//ix:nonFraction', ns):
            name = elem.get('name', '')
            if ':' in name:
                tag = name.split(':')[-1]
                for metric, patterns in METRICS.items():
                    if tag in patterns and metric not in data:
                        val = clean_number(elem.text)
                        if val is not None:
                            data[metric] = val
    except Exception:
        pass
    
    # Fallback without namespace
    for elem in root.iter():
        if 'nonFraction' in str(elem.tag):
            name = elem.get('name', '')
            if ':' in name:
                tag = name.split(':')[-1]
                for metric, patterns in METRICS.items():
                    if tag in patterns and metric not in data:
                        val = clean_number(elem.text)
                        if val is not None:
                            data[metric] = val

def strategy_2_old_xml(root, data):
    """Old XML structures"""
    for elem in root.iter():
        if '}' in elem.tag:
            tag = elem.tag.split('}')[1]
        else:
            tag = elem.tag
        for metric, patterns in METRICS.items():
            if tag in patterns and metric not in data:
                val = clean_number(elem.text)
                if val is not None:
                    data[metric] = val

def strategy_3_aggressive(root, data):
    """Aggressive tag search by lowering and matching"""
    for elem in root.iter():
        tag = elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag
        tag_lower = tag.lower()
        if tag_lower in TAG_TO_METRIC:
            metric = TAG_TO_METRIC[tag_lower]
            if metric not in data:
                val = clean_number(elem.text)
                if val is not None:
                    data[metric] = val

def strategy_4_regex(content_str, data):
    """Regex fallback on raw content string - OPTIMIZED"""
    for metric, pattern in COMPILED_PATTERNS['financial'].items():
        if metric not in data:
            match = pattern.search(content_str)
            if match:
                val = clean_number(match.group(1))
                if val is not None:
                    data[metric] = val

def strategy_5_ix_hidden(root, data):
    """Parse ix:hidden sections where iXBRL stores data"""
    namespaces = [
        {'ix': 'http://www.xbrl.org/2008/inlineXBRL'},
        {'ix': 'http://www.xbrl.org/2013/inlineXBRL'},
    ]
    for ns in namespaces:
        try:
            for hidden in root.findall('.//ix:hidden', ns):
                for elem in hidden.iter():
                    if 'nonFraction' in str(elem.tag):
                        name = elem.get('name', '')
                        if ':' in name:
                            tag = name.split(':')[-1]
                            for metric, patterns in METRICS.items():
                                if tag in patterns and metric not in data:
                                    val = clean_number(elem.text)
                                    if val is not None:
                                        data[metric] = val
        except Exception:
            continue
    
    # Also try without namespace prefix
    for elem in root.iter():
        tag_str = str(elem.tag).lower()
        if 'hidden' in tag_str:
            for child in elem.iter():
                if 'nonfraction' in str(child.tag).lower():
                    name = child.get('name', '')
                    if ':' in name:
                        tag = name.split(':')[-1]
                        for metric, patterns in METRICS.items():
                            if tag in patterns and metric not in data:
                                val = clean_number(child.text)
                                if val is not None:
                                    data[metric] = val

def strategy_6_ix_header(root, data):
    """Parse ix:header sections"""
    for elem in root.iter():
        tag_str = str(elem.tag).lower()
        if 'header' in tag_str:
            for child in elem.iter():
                child_tag = str(child.tag).lower()
                if 'nonfraction' in child_tag:
                    name = child.get('name', '')
                    if ':' in name:
                        tag = name.split(':')[-1]
                        for metric, patterns in METRICS.items():
                            if tag in patterns and metric not in data:
                                val = clean_number(child.text)
                                if val is not None:
                                    data[metric] = val

def strategy_7_name_attribute(root, data):
    """Search for name attributes containing metric names"""
    for elem in root.iter():
        name = elem.get('name', '')
        if not name:
            continue
        if ':' in name:
            tag = name.split(':')[-1]
        else:
            tag = name
        tag_lower = tag.lower()
        for metric, patterns in METRICS.items():
            for pattern in patterns:
                if pattern.lower() in tag_lower or tag_lower in pattern.lower():
                    if metric not in data:
                        val = clean_number(elem.text)
                        if val is not None:
                            data[metric] = val
                        break

def strategy_8_context_aware(root, content_str, data):
    """Try to use contextRef and pick current year numbers - OPTIMIZED"""
    if content_str is None:
        return
    contexts = COMPILED_PATTERNS['context'].findall(content_str)
    if contexts:
        main_context = contexts[0]
        pattern = re.compile(f'contextRef="{re.escape(main_context)}"[^>]*name="[^:]*:([^"]+)"[^>]*>([^<]*)<')
        matches = pattern.findall(content_str)
        for tag, value in matches:
            for metric, patterns in METRICS.items():
                if tag in patterns and metric not in data:
                    val = clean_number(value)
                    if val is not None:
                        data[metric] = val

def strategy_9_table_scraping(root, data):
    """Scrape HTML tables for labels and numbers"""
    for table in root.iter('table'):
        rows = table.findall('.//tr')
        for row in rows:
            cells = row.findall('.//td')
            if len(cells) >= 2:
                label_text = ' '.join(cells[0].itertext()).strip()
                label_lower = label_text.lower()
                for metric, patterns in METRICS.items():
                    if metric not in data:
                        for pattern in patterns:
                            if pattern.lower() in label_lower:
                                for cell in reversed(cells[1:]):
                                    cell_text = ' '.join(cell.itertext()).strip()
                                    val = clean_number(cell_text)
                                    if val is not None:
                                        data[metric] = val
                                        break
                                break

def strategy_10_all_namespaces(root, data):
    """Try a variety of namespace prefixes and ix:nonFraction matches"""
    prefixes = [
        'uk-gaap', 'ukGAAP', 'gaap', 'core', 
        'ns5', 'ns0', 'ns1', 'ns2', 'ns3', 'ns4', 'ns6', 'ns7', 'ns8', 'ns9', 'ns10',
        'uk-gaap-pt', 'uk-gaap-cd-bus', 'uk-gaap-rp-dir', 'FRS-102'
    ]
    for elem in root.iter():
        if 'nonFraction' in str(elem.tag).lower():
            name = elem.get('name', '')
            if ':' in name:
                tag = name.split(':')[-1]
                for metric, patterns in METRICS.items():
                    if tag in patterns and metric not in data:
                        val = clean_number(elem.text)
                        if val is not None:
                            data[metric] = val
    
    for prefix in prefixes:
        for metric, patterns in METRICS.items():
            if metric not in data:
                for pattern in patterns:
                    for elem in root.iter():
                        elem_name = elem.get('name', '')
                        if elem_name.endswith(pattern) or elem_name.endswith(f":{pattern}"):
                            val = clean_number(elem.text)
                            if val is not None:
                                data[metric] = val
                                break

def strategy_11_format_aware(root, data):
    """Handle ixt:zerodash / ixt2:zerodash / nocontent formats indicating zeros"""
    for elem in root.iter():
        format_attr = elem.get('format', '')
        if format_attr and ('zerodash' in format_attr.lower() or 'numdash' in format_attr.lower() or 'nocontent' in format_attr.lower()):
            name = elem.get('name', '')
            if ':' in name:
                tag = name.split(':')[-1]
                for metric, patterns in METRICS.items():
                    if tag in patterns and metric not in data:
                        text = (elem.text or '').strip()
                        if text in ['-', '–', '—', '', 'nil', 'Nil'] or not text:
                            data[metric] = 0.0
                        else:
                            val = clean_number(text)
                            if val is not None:
                                data[metric] = val

def strategy_12_llp_specific(root, data):
    """Handle LLP 'Members Interests' mapping to equity/net assets"""
    llp_mappings = {
        'equity': ['MembersInterests', 'TotalMembersInterests', 'MembersCapital'],
        'net_assets': ['NetAssetsLiabilitiesIncludingMembersInterests'],
    }
    for metric, patterns in llp_mappings.items():
        if metric not in data:
            for elem in root.iter():
                name = elem.get('name', '')
                if ':' in name:
                    tag = name.split(':')[-1]
                    if tag in patterns:
                        val = clean_number(elem.text)
                        if val is not None:
                            data[metric] = val
                            break

# ==========================================================================
# METADATA EXTRACTION (OPTIMIZED)
# ==========================================================================

def extract_metadata(content_str, filename, data):
    """Extract company_number and period_end - OPTIMIZED"""
    # Company number from filename (fastest)
    if 'company_number' not in data:
        match = COMPILED_PATTERNS['company_number_filename'].search(filename)
        if match:
            comp_num = match.group(1)
            data['company_number'] = comp_num if comp_num.startswith('OC') else comp_num.lstrip('0')
    
    # Period from filename
    if 'period_end' not in data:
        match = COMPILED_PATTERNS['period_end_filename'].search(filename)
        if match:
            d = match.group(1)
            data['period_end'] = f"{d[0:4]}-{d[4:6]}-{d[6:8]}"
    
    # If we got both from filename, return early
    if 'company_number' in data and 'period_end' in data:
        return
    
    # Otherwise search content (limit to first 10KB for speed)
    search_content = content_str[:10000]
    
    if 'company_number' not in data:
        for pattern in COMPILED_PATTERNS['company_number_xml']:
            match = pattern.search(search_content)
            if match:
                comp_num = match.group(1)
                data['company_number'] = comp_num if comp_num.startswith('OC') else comp_num.lstrip('0')
                break
    
    if 'period_end' not in data:
        for pattern in COMPILED_PATTERNS['period_end_xml']:
            match = pattern.search(search_content)
            if match:
                data['period_end'] = match.group(1)
                break

# ==========================================================================
# MAIN EXTRACTION FUNCTION
# ==========================================================================

def extract_with_all_strategies(content, filename):
    """Extract using ALL 12 strategies with optimizations"""
    if isinstance(content, bytes):
        content_str = content.decode('utf-8', errors='ignore')
    else:
        content_str = content
    
    data = {}
    
    # Extract metadata first (often from filename - very fast)
    extract_metadata(content_str, filename, data)
    
    # Try XML parsing
    root = None
    xml_parseable = False
    try:
        root = ET.fromstring(content_str)
        xml_parseable = True
        
        # Run all XML-based strategies
        strategy_1_ixbrl(root, data)
        strategy_2_old_xml(root, data)
        strategy_3_aggressive(root, data)
        strategy_5_ix_hidden(root, data)
        strategy_6_ix_header(root, data)
        strategy_7_name_attribute(root, data)
        strategy_8_context_aware(root, content_str, data)
        strategy_9_table_scraping(root, data)
        strategy_10_all_namespaces(root, data)
        strategy_11_format_aware(root, data)
        strategy_12_llp_specific(root, data)
        
    except Exception:
        xml_parseable = False
    
    # Strategy 4 (regex - always run as fallback)
    strategy_4_regex(content_str, data)
    
    # Check if we have enough data
    has_metadata = 'company_number' in data and 'period_end' in data
    financial = [k for k in data if k not in ['company_number', 'period_end']]
    has_financial = len(financial) >= 1
    
    if has_metadata or has_financial:
        return data
    else:
        return None

# ==========================================================================
# PROCESS ZIP WITH PARALLEL PROCESSING
# ==========================================================================

def process_single_file(args):
    """Process a single file - for parallel execution"""
    content, filename, year = args
    try:
        data = extract_with_all_strategies(content, filename)
        if data:
            data['year'] = year
            data['filename'] = filename
            return ('success', data)
        else:
            return ('failure', {'filename': filename, 'success': False})
    except Exception as e:
        return ('failure', {'filename': filename, 'error': str(e), 'success': False})

def process_zip_parallel(zip_path, year, sample_size=100, num_workers=4, save_failures=True):
    """Process ZIP with parallel execution and all strategies"""
    print(f"\n{'='*80}")
    print(f"{'TESTING' if TEST_MODE else 'PROCESSING'} {year}: {zip_path.name}")
    print(f"{'='*80}")
    
    companies = {}
    failure_logs = []
    successful_files = 0
    
    with zipfile.ZipFile(zip_path, 'r') as zf:
        all_files = [f for f in zf.namelist() if f.endswith(('.xml', '.html', '.htm', '.xhtml'))]
        
        if TEST_MODE and len(all_files) > sample_size:
            files = random.sample(all_files, sample_size)
            print(f"Testing {len(files)} random files (out of {len(all_files):,})")
        else:
            files = all_files
            print(f"Processing all {len(files):,} files")
        
        # Read all files first (fast disk I/O)
        file_data = []
        for filename in files:
            try:
                content = zf.read(filename)
                file_data.append((content, filename, year))
            except Exception as e:
                failure_logs.append({'filename': filename, 'error': f"Read error: {str(e)}"})
        
        # Process in parallel
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(process_single_file, args): args for args in file_data}
            
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"{year}", ncols=100):
                try:
                    status, result = future.result()
                    if status == 'success':
                        successful_files += 1
                        company_num = result.get('company_number', 'unknown')
                        companies[company_num] = result
                    else:
                        failure_logs.append(result)
                except Exception as e:
                    failure_logs.append({'error': str(e)})
    
    # Stats
    total = len(files)
    success = successful_files
    failed = total - success
    rate = (success / total * 100) if total > 0 else 0
    
    print(f"\n✓ Success: {success:,} / {total:,} ({rate:.1f}%)")
    print(f"  - Unique companies: {len(companies):,}")
    if successful_files > len(companies):
        print(f"  - Duplicate filings: {successful_files - len(companies):,}")
    print(f"✗ Failed: {failed:,} ({100-rate:.1f}%)")
    
    # Save failure details if requested
    if save_failures and failure_logs:
        log_dir = Path('test_failures')
        log_dir.mkdir(exist_ok=True)
        log_file = log_dir / f'failures_{year}.json'
        with open(log_file, 'w') as f:
            json.dump(failure_logs[:1000], f, indent=2)  # Limit to first 1000 to save space
        print(f"📋 Failure log: {log_file}")
    
    return companies, failure_logs, successful_files

# ==========================================================================
# BUILD TIMESERIES FROM CHECKPOINTS (MEMORY EFFICIENT)
# ==========================================================================

def build_timeseries_from_checkpoints(years):
    """Build timeseries by loading one checkpoint at a time"""
    print("\n" + "="*80)
    print("BUILDING TIME-SERIES FROM CHECKPOINTS")
    print("="*80)
    
    timeseries = defaultdict(list)
    
    for year in sorted(years):
        checkpoint_file = f'checkpoint_{year}.pkl'
        if not os.path.exists(checkpoint_file):
            print(f"⚠️  Missing checkpoint for {year}")
            continue
        
        print(f"Loading {year}...", end=' ')
        with open(checkpoint_file, 'rb') as f:
            cp = pickle.load(f)
        
        companies_dict = cp.get('companies', cp)
        
        for company_num, data in companies_dict.items():
            timeseries[company_num].append({'year': year, **data})
        
        print(f"✓ ({len(companies_dict):,} companies)")
    
    # Sort each company's timeline
    for company_num in timeseries:
        timeseries[company_num].sort(key=lambda x: x['year'])
    
    multi = {k: v for k, v in timeseries.items() if len(v) >= 2}
    single = {k: v[0] for k, v in timeseries.items() if len(v) == 1}
    
    print(f"\nUnique companies: {len(timeseries):,}")
    print(f"  Multi-year (2+): {len(multi):,}")
    print(f"  Single-year: {len(single):,}")
    
    if multi:
        counts = [len(v) for v in multi.values()]
        print(f"\nYears per company: Min={min(counts)}, Max={max(counts)}, Avg={sum(counts)/len(counts):.1f}")
    
    return multi, single

# ==========================================================================
# MAIN
# ==========================================================================

def main():
    print("\n" + "="*80)
    if TEST_MODE:
        print("OPTIMIZED PARSER - TEST MODE (All 12 Strategies)")
        print(f"Testing {SAMPLE_SIZE} random files per year")
    else:
        print("OPTIMIZED PARSER - FULL PROCESSING (All 12 Strategies)")
    print(f"Memory Light Mode: {MEMORY_LIGHT_MODE} ✅")
    print(f"Parallel Workers: {NUM_WORKERS}")
    print("="*80)

    DATA_DIR = Path(r'c:\\Users\\theoe\\OneDrive\\Documents\\GitHub\\manifold\\v1\\data\\CH\\April')

    ZIP_FILES = {
        2010: DATA_DIR / 'Accounts_Monthly_Data-April2010.zip',
        2011: DATA_DIR / 'Accounts_Monthly_Data-April2011.zip',
        2012: DATA_DIR / 'Accounts_Monthly_Data-April2012.zip',
        2013: DATA_DIR / 'Accounts_Monthly_Data-April2013.zip',
        2014: DATA_DIR / 'Accounts_Monthly_Data-April2014.zip',
        2015: DATA_DIR / 'Accounts_Monthly_Data-April2015.zip',
        2016: DATA_DIR / 'Accounts_Monthly_Data-April2016.zip',
        2017: DATA_DIR / 'Accounts_Monthly_Data-April2017.zip',
        2018: DATA_DIR / 'Accounts_Monthly_Data-April2018.zip',
        2019: DATA_DIR / 'Accounts_Monthly_Data-April2019.zip',
        2020: DATA_DIR / 'Accounts_Monthly_Data-April2020.zip',
        2021: DATA_DIR / 'Accounts_Monthly_Data-April2021.zip',
        2022: DATA_DIR / 'Accounts_Monthly_Data-April2022.zip',
        2023: DATA_DIR / 'Accounts_Monthly_Data-April2023.zip',
        2024: DATA_DIR / 'Accounts_Monthly_Data-April2024.zip',
    }
    
    available = {y: p for y, p in ZIP_FILES.items() if p.exists()}
    if not available:
        print("\n⨯ No files found")
        return
    
    print(f"\n📁 Found {len(available)} ZIP files")
    
    processed_years = []
    overall_successful_files = 0
    
    # Process each year (skipping if checkpoint exists)
    for year in sorted(available.keys()):
        checkpoint_file = f'checkpoint_{year}.pkl'
        
        if os.path.exists(checkpoint_file):
            print(f"\n✓ Skipping {year} (checkpoint exists)")
            processed_years.append(year)
            # Get stats from checkpoint for summary
            try:
                with open(checkpoint_file, 'rb') as f:
                    cp = pickle.load(f)
                successful_files = cp.get('successful_files', len(cp.get('companies', {})))
                overall_successful_files += successful_files
            except:
                pass
            continue
        
        companies, failures, successful_files = process_zip_parallel(
            available[year], 
            year, 
            sample_size=SAMPLE_SIZE,
            num_workers=NUM_WORKERS,
            save_failures=SAVE_FAILED_FILES
        )
        
        processed_years.append(year)
        overall_successful_files += successful_files
        
        # Save checkpoint
        cp_payload = {
            'companies': companies,
            'failures': failures,
            'successful_files': successful_files
        }
        
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(cp_payload, f)
        print(f"  ✓ Checkpoint saved: {checkpoint_file}")
    
    print("\n" + "="*80)
    print("PROCESSING COMPLETE")
    print("="*80)
    
    if not TEST_MODE:
        # Build timeseries from checkpoints (memory efficient)
        multi, single = build_timeseries_from_checkpoints(processed_years)
        
        print("\n" + "="*80)
        print("SAVING FINAL OUTPUT")
        print("="*80)
        
        # Save multi-year timeseries
        with open('companies_timeseries_15yr.pkl', 'wb') as f:
            pickle.dump(multi, f)
        print(f"✓ companies_timeseries_15yr.pkl: {len(multi):,} companies")
        
        # Create CSV
        rows = []
        for c, timeline in multi.items():
            for p in timeline:
                rows.append({'company_number': c, **p})
        df = pd.DataFrame(rows)
        df.to_csv('companies_timeseries_15yr.csv', index=False)
        print(f"✓ companies_timeseries_15yr.csv: {len(df):,} rows")
        
        print("\n🚀 Ready for Kalman filters!")
    else:
        print("\n💡 Set TEST_MODE=False to process all files and generate final output")

if __name__ == "__main__":
    main()