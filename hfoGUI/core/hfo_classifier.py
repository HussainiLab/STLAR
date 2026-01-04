"""
HFO Classification with co-occurrence detection.
Classifies ripples, fast ripples, and their co-occurrences.
"""

import numpy as np
from collections import Counter


class HFO_Classifier:
    """
    Classify HFO events including ripple-fast ripple co-occurrences.
    Follows epilepsy literature standards: at least 4 consecutive cycles,
    events within 25ms merged into single detection.
    """
    
    def __init__(self, fs=4800, cooccurrence_window_ms=25):
        self.fs = fs
        self.cooccurrence_window_ms = cooccurrence_window_ms
        self.window_samples = int(cooccurrence_window_ms / 1000 * fs)
    
    def classify_events(self, ripple_list, fr_list):
        """
        Classify ripples and fast ripples, detecting co-occurrences.
        
        Parameters
        ----------
        ripple_list : list of dicts
            Each dict has 'start_ms', 'end_ms', 'peak_freq'
        fr_list : list of dicts
            Each dict has 'start_ms', 'end_ms', 'peak_freq'
        
        Returns
        -------
        classified_events : list of dicts
            Sorted by start time. Each has 'type', 'is_cooccurrence', 'pathology_score'
        """
        
        events = []
        fr_used = set()  # Track which FRs are already paired
        
        # Process ripples first
        for r_idx, ripple in enumerate(ripple_list):
            r_start_ms = float(ripple.get('start_ms', 0))
            r_end_ms = float(ripple.get('end_ms', 0))
            
            # Check for FR co-occurrence (within window)
            fr_cooccurs = False
            cooccurring_fr_idx = None
            
            for fr_idx, fr in enumerate(fr_list):
                if fr_idx in fr_used:
                    continue  # Already paired
                
                fr_start_ms = float(fr.get('start_ms', 0))
                # Overlap or proximity check
                if abs(r_start_ms - fr_start_ms) < self.cooccurrence_window_ms:
                    fr_cooccurs = True
                    cooccurring_fr_idx = fr_idx
                    fr_used.add(fr_idx)
                    break
            
            # Classify
            if fr_cooccurs:
                event_type = 'ripple_fast_ripple'
                pathology_score = 4
            else:
                event_type = 'ripple'
                pathology_score = 2
            
            events.append({
                'start_ms': r_start_ms,
                'end_ms': r_end_ms,
                'type': event_type,
                'is_cooccurrence': fr_cooccurs,
                'pathology_score': pathology_score,
                'ripple_peak_freq': ripple.get('peak_freq'),
                'fr_peak_freq': fr_list[cooccurring_fr_idx].get('peak_freq') if cooccurring_fr_idx is not None else None
            })
        
        # Add isolated fast ripples
        for fr_idx, fr in enumerate(fr_list):
            if fr_idx in fr_used:
                continue
            
            event_type = 'fast_ripple'
            pathology_score = 3
            
            events.append({
                'start_ms': float(fr.get('start_ms', 0)),
                'end_ms': float(fr.get('end_ms', 0)),
                'type': event_type,
                'is_cooccurrence': False,
                'pathology_score': pathology_score,
                'ripple_peak_freq': None,
                'fr_peak_freq': fr.get('peak_freq')
            })
        
        # Sort by time
        events.sort(key=lambda x: x['start_ms'])
        
        return events
    
    def compute_summary(self, classified_events):
        """
        Compute summary statistics from classified events.
        
        Parameters
        ----------
        classified_events : list of dicts
            Output from classify_events()
        
        Returns
        -------
        summary : dict
            Keys: type_counts (dict), fr_to_ripple_ratio, cooccurrence_rate, etc.
        """
        
        type_counts = Counter(e['type'] for e in classified_events)
        
        total = len(classified_events)
        ripple_count = type_counts.get('ripple', 0) + type_counts.get('ripple_fast_ripple', 0)
        fr_only = type_counts.get('fast_ripple', 0)
        fr_cooccur = type_counts.get('ripple_fast_ripple', 0)
        cooccurrence_count = fr_cooccur
        
        summary = {
            'total_events': total,
            'pure_ripple': type_counts.get('ripple', 0),
            'pure_fast_ripple': fr_only,
            'ripple_fast_ripple_cooccurrence': fr_cooccur,
            
            # Ratios and rates
            'fr_to_ripple_ratio': (fr_only + fr_cooccur) / ripple_count if ripple_count > 0 else 0,
            'cooccurrence_rate': cooccurrence_count / total if total > 0 else 0,
            'mean_pathology_score': np.mean([e['pathology_score'] for e in classified_events]) if classified_events else 0
        }
        
        return summary
