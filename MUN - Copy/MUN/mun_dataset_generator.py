"""
MUN Conference Attendance Dataset Generator
Marathwada Region, Maharashtra, India

This script generates highly realistic synthetic data for predicting 
student attendance at Model UN conferences in the Marathwada region.

Author: Dataset Design for Educational Analytics
Region: Marathwada (Aurangabad, Jalna, Beed, Latur, Osmanabad, Nanded, Parbhani, Hingoli)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# ============================================================================
# MARATHWADA REGION CONFIGURATION
# ============================================================================

# Real cities in Marathwada with their characteristics
CITIES_CONFIG = {
    'Aurangabad': {
        'population': 1200000,
        'urban_tier': 2,
        'distance_to_venue': (5, 25),  # km range from typical venue
        'num_schools': 450,
        'mun_culture': 'high',
        'avg_income': 'mid-high',
        'transport_quality': 'good',
        'districts': ['Aurangabad City', 'Waluj', 'Paithan', 'Gangapur']
    },
    'Jalna': {
        'population': 300000,
        'urban_tier': 3,
        'distance_to_venue': (20, 35),
        'num_schools': 180,
        'mun_culture': 'medium',
        'avg_income': 'mid',
        'transport_quality': 'average',
        'districts': ['Jalna City', 'Bhokardan', 'Jafrabad']
    },
    'Beed': {
        'population': 150000,
        'urban_tier': 3,
        'distance_to_venue': (45, 65),
        'num_schools': 120,
        'mun_culture': 'low-medium',
        'avg_income': 'low-mid',
        'transport_quality': 'average',
        'districts': ['Beed City', 'Parli', 'Ashti']
    },
    'Latur': {
        'population': 400000,
        'urban_tier': 2,
        'distance_to_venue': (55, 75),
        'num_schools': 250,
        'mun_culture': 'medium',
        'avg_income': 'mid',
        'transport_quality': 'average',
        'districts': ['Latur City', 'Chakur', 'Ausa']
    },
    'Osmanabad': {
        'population': 120000,
        'urban_tier': 3,
        'distance_to_venue': (75, 95),
        'num_schools': 95,
        'mun_culture': 'low',
        'avg_income': 'low-mid',
        'transport_quality': 'poor-average',
        'districts': ['Osmanabad City', 'Tuljapur', 'Omerga']
    },
    'Nanded': {
        'population': 550000,
        'urban_tier': 2,
        'distance_to_venue': (30, 50),
        'num_schools': 280,
        'mun_culture': 'medium',
        'avg_income': 'mid',
        'transport_quality': 'average',
        'districts': ['Nanded City', 'Deglur', 'Mukhed']
    },
    'Parbhani': {
        'population': 280000,
        'urban_tier': 3,
        'distance_to_venue': (70, 90),
        'num_schools': 150,
        'mun_culture': 'low-medium',
        'avg_income': 'low-mid',
        'transport_quality': 'poor-average',
        'districts': ['Parbhani City', 'Purna', 'Jintur']
    },
    'Hingoli': {
        'population': 85000,
        'urban_tier': 4,
        'distance_to_venue': (85, 110),
        'num_schools': 70,
        'mun_culture': 'low',
        'avg_income': 'low',
        'transport_quality': 'poor',
        'districts': ['Hingoli City', 'Kalamnuri', 'Basmath']
    }
}

# Real and realistic school names in Marathwada
SCHOOLS = {
    'Aurangabad': [
        'Nath Valley School', 'Delhi Public School Aurangabad', 'Millenium National School',
        'St. Francis De Sales High School', 'Cambridge Court High School', 'MGM High School',
        'Stepping Stones High School', 'Indira National School', 'Jalna Road High School',
        'Vasantrao Naik High School', 'New English School', 'Modern English School',
        'Zilla Parishad High School Waluj', 'Dr. Babasaheb Ambedkar School',
        'Maulana Azad High School', 'Saraswati Vidyalaya', 'Bharati Vidyapeeth School'
    ],
    'Jalna': [
        'Jalna Public School', 'New English High School Jalna', 'Shivaji Vidyalaya',
        'Vidya Vikas High School', 'Sant Gadge Baba School', 'Z.P. High School Jalna',
        'Marathi Vidyalaya', 'Dnyandeep Convent', 'Ideal English School'
    ],
    'Beed': [
        'Vasantrao Naik Vidyalaya Beed', 'New English School Beed', 'Shivaji High School',
        'Z.P. School Beed', 'Chatrapati Shahu School', 'Parli Public School',
        'Modern High School Beed', 'Ashti Vidyalaya'
    ],
    'Latur': [
        'Radhabai Kale School', 'Cambridge School Latur', 'Shivaji Vidyalaya Latur',
        'New English School Latur', 'Z.P. High School Latur', 'Vasantrao Naik School',
        'Modern English Medium School', 'Saraswati Vidya Mandir', 'Apex School Latur'
    ],
    'Osmanabad': [
        'Osmanabad Public School', 'New English School Osmanabad', 'Tuljapur Vidyalaya',
        'Z.P. High School Osmanabad', 'Marathi Shala Osmanabad', 'Dnyaneshwar School'
    ],
    'Nanded': [
        'Nanded Public School', 'Delhi Public School Nanded', 'Cambridge International School',
        'New English School Nanded', 'Z.P. School Nanded', 'Shivaji Vidyalaya Nanded',
        'Gurunanak Vidyalaya', 'Maratha Vidya Prasarak School'
    ],
    'Parbhani': [
        'Parbhani High School', 'New English School Parbhani', 'Z.P. School Parbhani',
        'Shivaji Vidyalaya Parbhani', 'Dnyandeep School', 'Purna Vidyalaya'
    ],
    'Hingoli': [
        'Hingoli Public School', 'Z.P. High School Hingoli', 'New English School Hingoli',
        'Kalamnuri Vidyalaya', 'Basmath High School'
    ]
}

# School types with realistic distribution
SCHOOL_TYPES = {
    'private': 0.35,      # 35% private (English medium, better facilities)
    'semi-private': 0.25, # 25% semi-private (aided schools)
    'government': 0.40    # 40% government (ZP, municipal schools)
}

# Transport modes based on distance and city
TRANSPORT_MODES = ['bus', 'train', 'car', 'auto', 'shared_taxi']

# Academic calendar considerations (Maharashtra board)
EXAM_PERIODS = [
    (datetime(2024, 10, 1), datetime(2024, 10, 15)),   # Unit tests
    (datetime(2024, 11, 15), datetime(2024, 11, 30)),  # Mid-term
    (datetime(2024, 12, 15), datetime(2025, 1, 5)),    # Pre-board prep
    (datetime(2025, 2, 15), datetime(2025, 3, 31)),    # Board exams
]

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def select_city_weighted():
    """Select city based on population and MUN culture"""
    cities = list(CITIES_CONFIG.keys())
    weights = [CITIES_CONFIG[c]['population'] for c in cities]
    return random.choices(cities, weights=weights, k=1)[0]

def get_school(city, school_type):
    """Get realistic school name based on city and type"""
    school_pool = SCHOOLS[city]
    
    if school_type == 'government':
        # Prefer schools with Z.P., Municipal, or Marathi in name
        govt_schools = [s for s in school_pool if any(x in s for x in ['Z.P.', 'Zilla', 'Marathi', 'Dr. Babasaheb'])]
        if govt_schools:
            return random.choice(govt_schools)
    elif school_type == 'private':
        # Prefer English medium, DPS, Cambridge, International
        private_schools = [s for s in school_pool if any(x in s for x in ['DPS', 'Cambridge', 'International', 'Public', 'Convent'])]
        if private_schools:
            return random.choice(private_schools)
    
    return random.choice(school_pool)

def calculate_travel_time(distance_km, transport_mode, transport_quality):
    """Calculate realistic travel time based on Marathwada road conditions"""
    # Base speed (km/hr) - Marathwada roads are often not great
    base_speeds = {
        'car': 50,           # Highway conditions
        'bus': 40,           # State transport buses
        'train': 45,         # Local trains are slow
        'auto': 35,          # Auto rickshaws
        'shared_taxi': 48    # Shared cabs
    }
    
    # Quality multiplier
    quality_multipliers = {
        'good': 1.0,
        'average': 0.8,
        'poor': 0.6
    }
    
    speed = base_speeds.get(transport_mode, 40) * quality_multipliers.get(transport_quality, 0.8)
    travel_time = distance_km / speed
    
    # Add waiting/boarding time
    waiting_time = {
        'car': 0.1,
        'auto': 0.15,
        'bus': 0.5,
        'train': 0.7,
        'shared_taxi': 0.3
    }
    
    return round(travel_time + waiting_time.get(transport_mode, 0.3), 2)

def get_income_group(city, school_type):
    """Determine income group based on city and school type"""
    city_income_map = {
        'mid-high': {'private': 0.7, 'semi-private': 0.25, 'government': 0.05},
        'mid': {'private': 0.5, 'semi-private': 0.35, 'government': 0.15},
        'low-mid': {'private': 0.3, 'semi-private': 0.35, 'government': 0.35},
        'low': {'private': 0.15, 'semi-private': 0.25, 'government': 0.60}
    }
    
    city_avg = CITIES_CONFIG[city]['avg_income']
    
    if school_type == 'private':
        return np.random.choice(['high', 'mid', 'low'], p=[0.5, 0.4, 0.1])
    elif school_type == 'semi-private':
        return np.random.choice(['high', 'mid', 'low'], p=[0.1, 0.6, 0.3])
    else:  # government
        return np.random.choice(['high', 'mid', 'low'], p=[0.05, 0.35, 0.6])

def get_parent_education(income_group, city):
    """Parent education level - realistic for Marathwada"""
    tier = CITIES_CONFIG[city]['urban_tier']
    
    # Higher tier = better education levels
    if tier == 2:
        education_probs = {
            'high': [0.4, 0.5, 0.1],    # [high, mid, low]
            'mid': [0.2, 0.6, 0.2],
            'low': [0.05, 0.4, 0.55]
        }
    else:  # tier 3-4
        education_probs = {
            'high': [0.3, 0.5, 0.2],
            'mid': [0.1, 0.5, 0.4],
            'low': [0.05, 0.3, 0.65]
        }
    
    return np.random.choice(['high', 'mid', 'low'], p=education_probs[income_group])

def calculate_performance_score(awards):
    """Calculate weighted performance score"""
    score = (
        3 * awards['best_delegate'] +
        2 * awards['high_commendation'] +
        1.5 * awards['special_mention'] +
        1 * awards['verbal_mention'] +
        4 * awards['best_delegation']
    )
    return round(score, 2)

def get_experience_level(num_conferences):
    """Categorize experience level"""
    if num_conferences == 0:
        return 'beginner'
    elif num_conferences <= 3:
        return 'beginner'
    elif num_conferences <= 7:
        return 'intermediate'
    else:
        return 'advanced'

def calculate_commitment_score(attendance_rate, interest, payment_status):
    """Calculate commitment score"""
    payment_mult = 1.0 if payment_status == 'paid' else 0.5
    return round(attendance_rate * (interest / 10) * payment_mult, 3)

def days_to_nearest_exam(registration_date):
    """Calculate days to nearest exam period"""
    min_days = 365
    for exam_start, exam_end in EXAM_PERIODS:
        days_to_start = abs((exam_start - registration_date).days)
        days_to_end = abs((exam_end - registration_date).days)
        min_days = min(min_days, days_to_start, days_to_end)
    return min_days

def get_transport_availability(distance, city):
    """Realistic transport availability based on distance and city"""
    base_quality = CITIES_CONFIG[city]['transport_quality']
    
    if distance < 30:
        quality_map = {'good': 'good', 'average': 'good', 'poor': 'average', 'poor-average': 'average'}
        return quality_map.get(base_quality, 'good')
    elif distance < 60:
        quality_map = {'good': 'average', 'average': 'average', 'poor': 'poor', 'poor-average': 'poor'}
        return quality_map.get(base_quality, 'average')
    else:
        quality_map = {'good': 'average', 'average': 'poor', 'poor': 'poor', 'poor-average': 'poor'}
        return quality_map.get(base_quality, 'poor')

def get_transport_mode(distance, income_group, transport_availability):
    """Select realistic transport mode"""
    if distance < 15:
        # Short distance - auto, car
        if income_group == 'high':
            return np.random.choice(['car', 'auto'], p=[0.7, 0.3])
        else:
            return np.random.choice(['auto', 'bus', 'car'], p=[0.5, 0.3, 0.2])
    
    elif distance < 50:
        # Medium distance
        if income_group == 'high':
            return np.random.choice(['car', 'shared_taxi', 'bus'], p=[0.6, 0.25, 0.15])
        elif income_group == 'mid':
            return np.random.choice(['bus', 'shared_taxi', 'car'], p=[0.5, 0.3, 0.2])
        else:
            return np.random.choice(['bus', 'shared_taxi'], p=[0.7, 0.3])
    
    else:
        # Long distance
        if income_group == 'high':
            return np.random.choice(['car', 'train', 'bus'], p=[0.5, 0.3, 0.2])
        elif income_group == 'mid':
            return np.random.choice(['train', 'bus', 'shared_taxi'], p=[0.4, 0.4, 0.2])
        else:
            return np.random.choice(['bus', 'train'], p=[0.6, 0.4])

# ============================================================================
# MAIN DATASET GENERATION FUNCTION
# ============================================================================

def generate_mun_dataset(num_samples=1000, conference_date='2024-11-10'):
    """
    Generate realistic MUN attendance dataset for Marathwada region
    
    Parameters:
    -----------
    num_samples : int
        Number of delegate records to generate
    conference_date : str
        Date of the conference (YYYY-MM-DD format)
    
    Returns:
    --------
    pandas.DataFrame
        Complete dataset with all features
    """
    
    conference_dt = datetime.strptime(conference_date, '%Y-%m-%d')
    data = []
    
    print(f"🎯 Generating {num_samples} realistic MUN delegate records...")
    print(f"📍 Region: Marathwada, Maharashtra")
    print(f"📅 Conference Date: {conference_date}")
    print("=" * 70)
    
    for i in range(num_samples):
        if (i + 1) % 200 == 0:
            print(f"✓ Generated {i + 1} records...")
        
        # ========== 1. BASIC PROFILE ==========
        delegate_id = f"MUN2024_{i+1:04d}"
        age = np.random.choice(range(9, 17), p=[0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.15, 0.08])
        class_num = max(4, min(10, age - 5))  # Realistic class based on age
        gender = np.random.choice(['Male', 'Female', 'Other'], p=[0.52, 0.47, 0.01])
        
        # ========== 2. LOCATION ==========
        city = select_city_weighted()
        city_config = CITIES_CONFIG[city]
        district = np.random.choice(city_config['districts'])
        
        # School selection
        school_type = np.random.choice(
            list(SCHOOL_TYPES.keys()),
            p=list(SCHOOL_TYPES.values())
        )
        school_name = get_school(city, school_type)
        
        # Distance calculation
        distance_min, distance_max = city_config['distance_to_venue']
        distance_to_venue = round(np.random.uniform(distance_min, distance_max), 1)
        
        # ========== 3. SOCIO-ECONOMIC ==========
        income_group = get_income_group(city, school_type)
        parent_education = get_parent_education(income_group, city)
        
        # ========== 4. TRANSPORT ==========
        transport_availability = get_transport_availability(distance_to_venue, city)
        transport_mode = get_transport_mode(distance_to_venue, income_group, transport_availability)
        travel_time = calculate_travel_time(distance_to_venue, transport_mode, transport_availability)
        
        # Parent travel concern based on distance and transport
        if distance_to_venue < 30 and transport_availability == 'good':
            parent_travel_concern = np.random.choice(['low', 'medium', 'high'], p=[0.7, 0.25, 0.05])
        elif distance_to_venue > 70 or transport_availability == 'poor':
            parent_travel_concern = np.random.choice(['low', 'medium', 'high'], p=[0.1, 0.3, 0.6])
        else:
            parent_travel_concern = np.random.choice(['low', 'medium', 'high'], p=[0.3, 0.5, 0.2])
        
        # ========== 5. MUN EXPERIENCE ==========
        # Experience depends on city MUN culture, school type, and age
        mun_culture = city_config['mun_culture']
        
        if mun_culture == 'high' and school_type == 'private':
            num_conferences = np.random.choice(range(0, 16), p=[0.05, 0.10, 0.10, 0.12, 0.12, 0.11, 0.10, 0.08, 0.07, 0.05, 0.04, 0.03, 0.02, 0.01, 0.00, 0.00])
        elif mun_culture in ['medium', 'low-medium']:
            num_conferences = np.random.choice(range(0, 10), p=[0.25, 0.20, 0.15, 0.12, 0.10, 0.08, 0.05, 0.03, 0.015, 0.005])
        else:
            num_conferences = np.random.choice(range(0, 6), p=[0.50, 0.25, 0.12, 0.08, 0.04, 0.01])
        
        # Awards (only if attended conferences)
        if num_conferences > 0:
            # Award probability increases with experience
            award_prob = min(0.35, num_conferences * 0.05)
            
            num_best_delegate = np.random.binomial(max(1, num_conferences // 4), award_prob * 0.4)
            num_high_commendation = np.random.binomial(max(1, num_conferences // 3), award_prob * 0.6)
            num_special_mention = np.random.binomial(max(1, num_conferences // 2), award_prob * 0.8)
            num_verbal_mention = np.random.binomial(num_conferences, award_prob * 1.0)
            num_best_delegation = np.random.binomial(max(1, num_conferences // 5), award_prob * 0.3)
        else:
            num_best_delegate = num_high_commendation = num_special_mention = 0
            num_verbal_mention = num_best_delegation = 0
        
        awards = {
            'best_delegate': num_best_delegate,
            'high_commendation': num_high_commendation,
            'special_mention': num_special_mention,
            'verbal_mention': num_verbal_mention,
            'best_delegation': num_best_delegation
        }
        performance_score = calculate_performance_score(awards)
        
        # Last conference gap
        if num_conferences > 0:
            last_conference_gap = np.random.choice([15, 30, 60, 90, 120, 180, 270, 365],
                                                   p=[0.15, 0.25, 0.2, 0.15, 0.1, 0.08, 0.05, 0.02])
        else:
            last_conference_gap = 0
        
        experience_level = get_experience_level(num_conferences)
        
        # ========== 6. SKILLS & INTEREST ==========
        # Interest correlated with experience and awards
        base_interest = 3 + min(3, num_conferences * 0.5) + min(2, performance_score * 0.1)
        interest_level = int(min(10, max(1, np.random.normal(base_interest, 1.5))))
        
        # Confidence grows with experience
        base_confidence = 3 + min(4, num_conferences * 0.4) + (1 if num_best_delegate > 0 else 0)
        confidence_level = int(min(10, max(1, np.random.normal(base_confidence, 1.2))))
        
        # Skills correlate with experience and school type
        skill_boost = 1 if school_type == 'private' else 0.5 if school_type == 'semi-private' else 0
        public_speaking_skill = int(min(10, max(1, np.random.normal(4 + num_conferences * 0.3 + skill_boost, 1.5))))
        research_skill = int(min(10, max(1, np.random.normal(4 + num_conferences * 0.25 + skill_boost, 1.5))))
        
        # ========== 7. SOCIAL FACTORS ==========
        # Friend attending - more likely if from same school/city
        if city in ['Aurangabad', 'Nanded', 'Latur']:
            friend_attending = np.random.choice([0, 1], p=[0.4, 0.6])
        else:
            friend_attending = np.random.choice([0, 1], p=[0.65, 0.35])
        
        # Peer influence
        if friend_attending == 1:
            peer_influence = np.random.choice(['low', 'medium', 'high'], p=[0.1, 0.3, 0.6])
        else:
            peer_influence = np.random.choice(['low', 'medium', 'high'], p=[0.5, 0.35, 0.15])
        
        # Social media engagement - correlated with age and school type
        if age >= 14 and school_type == 'private':
            social_media = np.random.choice(['low', 'medium', 'high'], p=[0.15, 0.35, 0.5])
        elif age >= 13:
            social_media = np.random.choice(['low', 'medium', 'high'], p=[0.3, 0.45, 0.25])
        else:
            social_media = np.random.choice(['low', 'medium', 'high'], p=[0.5, 0.35, 0.15])
        
        # ========== 8. REGISTRATION BEHAVIOR ==========
        # Registration timing - experienced delegates register earlier
        if num_conferences >= 5:
            reg_days_before = int(np.random.gamma(4, 5) + 15)  # Early registration
        elif num_conferences >= 2:
            reg_days_before = int(np.random.gamma(3, 4) + 10)  # Regular
        else:
            reg_days_before = int(np.random.gamma(2, 3) + 5)   # Late registration
        
        reg_days_before = min(60, max(1, reg_days_before))
        
        if reg_days_before >= 25:
            registration_type = 'early'
        elif reg_days_before >= 12:
            registration_type = 'regular'
        else:
            registration_type = 'late'
        
        # Payment status - correlated with income and registration timing
        if income_group == 'high' and reg_days_before > 15:
            payment_status = np.random.choice(['paid', 'pending'], p=[0.85, 0.15])
        elif income_group == 'mid':
            payment_status = np.random.choice(['paid', 'pending'], p=[0.65, 0.35])
        else:
            payment_status = np.random.choice(['paid', 'pending'], p=[0.45, 0.55])
        
        # Form completion quality
        if school_type == 'private' and age >= 13:
            form_quality = np.random.choice(['low', 'medium', 'high'], p=[0.05, 0.25, 0.7])
        elif school_type == 'semi-private':
            form_quality = np.random.choice(['low', 'medium', 'high'], p=[0.15, 0.5, 0.35])
        else:
            form_quality = np.random.choice(['low', 'medium', 'high'], p=[0.3, 0.5, 0.2])
        
        # ========== 9. SCHOOL & FAMILY SUPPORT ==========
        # School support based on type and MUN culture
        if school_type == 'private' and mun_culture == 'high':
            school_support = np.random.choice(['low', 'medium', 'high'], p=[0.05, 0.25, 0.7])
        elif school_type == 'private':
            school_support = np.random.choice(['low', 'medium', 'high'], p=[0.1, 0.4, 0.5])
        elif mun_culture in ['medium', 'low-medium']:
            school_support = np.random.choice(['low', 'medium', 'high'], p=[0.25, 0.5, 0.25])
        else:
            school_support = np.random.choice(['low', 'medium', 'high'], p=[0.45, 0.4, 0.15])
        
        # Teacher encouragement (1-10 scale)
        teacher_encouragement = int(np.random.normal(
            {'low': 4, 'medium': 6, 'high': 8}[school_support], 1.5
        ))
        teacher_encouragement = max(1, min(10, teacher_encouragement))
        
        # Parent support - depends on education, income, and distance
        if parent_education == 'high' and distance_to_venue < 40:
            parent_support = np.random.choice(['yes', 'no'], p=[0.85, 0.15])
        elif parent_education == 'mid' and income_group in ['mid', 'high']:
            parent_support = np.random.choice(['yes', 'no'], p=[0.65, 0.35])
        elif distance_to_venue > 70:
            parent_support = np.random.choice(['yes', 'no'], p=[0.4, 0.6])
        else:
            parent_support = np.random.choice(['yes', 'no'], p=[0.55, 0.45])
        
        # ========== 10. FINANCIAL FACTORS ==========
        can_afford_travel = 'yes' if income_group in ['high', 'mid'] or distance_to_venue < 30 else \
                           np.random.choice(['yes', 'no'], p=[0.6, 0.4])
        
        sponsorship_needed = 'yes' if income_group == 'low' and distance_to_venue > 50 else \
                            np.random.choice(['yes', 'no'], p=[0.2, 0.8])
        
        # ========== 11. EXTERNAL FACTORS ==========
        registration_date = conference_dt - timedelta(days=reg_days_before)
        exam_proximity_days = days_to_nearest_exam(registration_date)
        
        if exam_proximity_days < 10:
            academic_pressure = 'high'
        elif exam_proximity_days < 25:
            academic_pressure = 'medium'
        else:
            academic_pressure = 'low'
        
        # Weather risk (Marathwada - monsoon Jun-Sep, winter Nov-Feb)
        conf_month = conference_dt.month
        if conf_month in [7, 8]:  # Peak monsoon
            weather_risk = np.random.choice(['low', 'medium', 'high'], p=[0.2, 0.3, 0.5])
        elif conf_month in [6, 9]:  # Monsoon onset/end
            weather_risk = np.random.choice(['low', 'medium', 'high'], p=[0.3, 0.5, 0.2])
        else:
            weather_risk = np.random.choice(['low', 'medium', 'high'], p=[0.7, 0.25, 0.05])
        
        health_issues = np.random.choice(['yes', 'no'], p=[0.08, 0.92])
        
        # ========== 12. HISTORICAL RELIABILITY ==========
        previous_registrations = min(num_conferences + np.random.randint(0, 3), num_conferences + 2)
        
        if num_conferences > 0:
            # Attendance rate based on experience and performance
            base_rate = 0.5 + min(0.3, num_conferences * 0.04) + min(0.15, performance_score * 0.01)
            previous_attendance_rate = round(min(1.0, max(0.1, np.random.normal(base_rate, 0.15))), 3)
        else:
            previous_attendance_rate = 0.0
        
        last_event_attended = 1 if num_conferences > 0 and previous_attendance_rate > 0.5 else \
                             np.random.choice([0, 1], p=[0.6, 0.4])
        
        dropout_history = max(0, previous_registrations - num_conferences)
        
        # ========== 13. DERIVED FEATURES ==========
        commitment_score = calculate_commitment_score(previous_attendance_rate, interest_level, payment_status)
        
        # Distance penalty (non-linear)
        if distance_to_venue < 30:
            distance_penalty = round(distance_to_venue * 0.01, 3)
        elif distance_to_venue < 60:
            distance_penalty = round(0.3 + (distance_to_venue - 30) * 0.015, 3)
        else:
            distance_penalty = round(0.75 + (distance_to_venue - 60) * 0.02, 3)
        
        # Experience score
        experience_score = round(num_conferences * 0.5 + performance_score * 0.3, 2)
        
        # Logistical difficulty (composite)
        logistics_factors = {
            'good': 0.1, 'average': 0.5, 'poor': 0.9,
            'low': 0.1, 'medium': 0.5, 'high': 0.9
        }
        logistical_difficulty = round(
            (distance_penalty + 
             logistics_factors.get(transport_availability, 0.5) +
             logistics_factors.get(parent_travel_concern, 0.5)) / 3, 3
        )
        
        # ========== 14. TARGET VARIABLE (WILL_ATTEND) ==========
        # Complex decision function with realistic correlations
        
        # Positive factors
        score = 0.0
        score += previous_attendance_rate * 25  # Strongest predictor
        score += 15 if payment_status == 'paid' else 0
        score += interest_level * 1.5
        score += confidence_level * 1.2
        score += performance_score * 0.8
        score += 8 if parent_support == 'yes' else -5
        score += {'high': 6, 'medium': 3, 'low': 0}[school_support]
        score += 5 if friend_attending == 1 else 0
        score += teacher_encouragement * 0.6
        score += {'early': 5, 'regular': 2, 'late': -2}[registration_type]
        score += {'high': 3, 'medium': 1, 'low': 0}[form_quality]
        score += experience_score * 0.5
        
        # Negative factors
        score -= distance_penalty * 15
        score -= logistical_difficulty * 8
        score -= {'high': 8, 'medium': 4, 'low': 1}[parent_travel_concern]
        score -= {'poor': 8, 'average': 3, 'good': 0}[transport_availability]
        score -= dropout_history * 5
        score -= max(0, (15 - exam_proximity_days)) * 0.8  # Exam pressure
        score -= {'high': 6, 'medium': 3, 'low': 0}[academic_pressure]
        score -= 3 if health_issues == 'yes' else 0
        score -= 4 if can_afford_travel == 'no' else 0
        score -= 3 if sponsorship_needed == 'yes' else 0
        score -= {'high': 4, 'medium': 2, 'low': 0}[weather_risk]
        
        # Income and education influence
        score += {'high': 3, 'mid': 1, 'low': -2}[income_group]
        score += {'high': 2, 'mid': 0, 'low': -1}[parent_education]
        
        # Convert to probability
        probability = 1 / (1 + np.exp(-score / 15))  # Sigmoid
        
        # Add some randomness (±10%)
        probability = max(0.05, min(0.95, probability + np.random.normal(0, 0.1)))
        
        will_attend = 1 if np.random.random() < probability else 0
        
        # ========== COMPILE RECORD ==========
        record = {
            # Basic Profile
            'delegate_id': delegate_id,
            'age': age,
            'class': class_num,
            'gender': gender,
            'city': city,
            'district': district,
            'school_name': school_name,
            'school_type': school_type,
            
            # Location & Travel
            'distance_to_venue_km': distance_to_venue,
            'travel_time_hours': travel_time,
            'transport_mode': transport_mode,
            'transport_availability': transport_availability,
            'parent_travel_concern': parent_travel_concern,
            
            # MUN Experience
            'num_conferences': num_conferences,
            'num_best_delegate_awards': num_best_delegate,
            'num_high_commendations': num_high_commendation,
            'num_special_mentions': num_special_mention,
            'num_verbal_mentions': num_verbal_mention,
            'num_best_delegation_awards': num_best_delegation,
            'last_conference_gap_days': last_conference_gap,
            'experience_level': experience_level,
            'performance_score': performance_score,
            
            # Behavior & Interest
            'interest_level': interest_level,
            'confidence_level': confidence_level,
            'public_speaking_skill': public_speaking_skill,
            'research_skill': research_skill,
            'peer_influence': peer_influence,
            'friend_attending': friend_attending,
            'social_media_engagement': social_media,
            
            # Registration
            'registration_time_days_before': reg_days_before,
            'registration_type': registration_type,
            'payment_status': payment_status,
            'form_completion_quality': form_quality,
            
            # School & Family
            'school_support_level': school_support,
            'teacher_encouragement': teacher_encouragement,
            'parent_support': parent_support,
            'parent_education_level': parent_education,
            
            # Socio-Economic
            'income_group': income_group,
            'can_afford_travel': can_afford_travel,
            'sponsorship_needed': sponsorship_needed,
            
            # External Factors
            'exam_proximity_days': exam_proximity_days,
            'academic_pressure': academic_pressure,
            'weather_risk': weather_risk,
            'health_issues_recent': health_issues,
            
            # Historical
            'previous_registrations': previous_registrations,
            'previous_attendance_rate': previous_attendance_rate,
            'last_event_attended': last_event_attended,
            'dropout_history': dropout_history,
            
            # Derived Features
            'commitment_score': commitment_score,
            'distance_penalty': distance_penalty,
            'experience_score': experience_score,
            'logistical_difficulty_score': logistical_difficulty,
            
            # Target
            'will_attend': will_attend
        }
        
        data.append(record)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    print(f"\n{'='*70}")
    print(f"✅ Dataset generation complete!")
    print(f"📊 Total records: {len(df)}")
    print(f"📈 Attendance rate: {df['will_attend'].mean()*100:.1f}%")
    print(f"\n🏙️  City Distribution:")
    print(df['city'].value_counts())
    print(f"\n🎓 School Type Distribution:")
    print(df['school_type'].value_counts())
    print(f"\n💼 Income Group Distribution:")
    print(df['income_group'].value_counts())
    
    return df

# ============================================================================
# EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Generate dataset
    dataset = generate_mun_dataset(num_samples=1500, conference_date='2024-11-10')
    
    # Save to CSV
    output_file = '/home/claude/mun_marathwada_dataset.csv'
    dataset.to_csv(output_file, index=False)
    print(f"\n💾 Dataset saved to: {output_file}")
    
    # Display sample
    print(f"\n📋 Sample Records (first 5):")
    print(dataset.head())
    
    print(f"\n📊 Dataset Statistics:")
    print(dataset.describe())
    
    print(f"\n🎯 Feature Summary:")
    print(f"Total Features: {len(dataset.columns)}")
    print(f"Numerical Features: {len(dataset.select_dtypes(include=[np.number]).columns)}")
    print(f"Categorical Features: {len(dataset.select_dtypes(include=['object']).columns)}")
    
    print("\n✨ Dataset generation script completed successfully!")
