#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script for Layer 1-3 causal skill profiling architecture
"""

import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
from config_loader import get_config
from gnn_talent_analyzer import TalentAnalyzer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_mock_data(n_members=50, n_excellent=5):
    """Create mock data for testing with exact column names from config"""

    # Get column names from config
    col_member_code = get_config('column_names.member.code', 'メンバーコード  ###[member_code]###')
    col_member_name = get_config('column_names.member.name', 'メンバー名  ###[name]###')
    col_retired_day = get_config('column_names.member.retired_day', '退職年月日  ###[retired_day]###')
    col_enter_day = get_config('column_names.member.enter_day', '入社年月日  ###[enter_day]###')
    col_job_grade = get_config('column_names.member.job_grade', '職能・等級  ###[job_grade]###')
    col_job_position = get_config('column_names.member.job_position', '役職  ###[job_position]###')

    col_acquired_member = get_config('column_names.acquired.member_code', 'メンバーコード')
    col_competence_type = get_config('column_names.acquired.competence_type', '力量タイプ  ###[competence_type]###')
    col_competence_code = get_config('column_names.acquired.competence_code', '力量コード')
    col_competence_name = get_config('column_names.acquired.competence_name', '力量名')
    col_level = get_config('column_names.acquired.level', 'レベル')

    col_skill_type = get_config('column_names.acquired.competence_types.skill', 'SKILL')

    # Create member dataframe with exact column names
    member_codes = [f'M{i:03d}' for i in range(n_members)]
    base_date = datetime(2020, 1, 1)

    member_df = pd.DataFrame({
        col_member_code: member_codes,
        col_member_name: [f'Member_{i}' for i in range(n_members)],
        col_retired_day: [None] * n_members,  # All active (not retired)
        col_enter_day: [base_date + timedelta(days=i*10) for i in range(n_members)],
        col_job_grade: np.random.choice(['E1', 'E2', 'E3', 'E4'], n_members),
        col_job_position: np.random.choice(['部長', '課長', '係長'], n_members),
    })

    # Create skill acquisition dataframe with exact column names
    n_skills = 15
    skill_codes = [f'SK{i:03d}' for i in range(n_skills)]

    rows = []
    for member_code in member_codes:
        for skill_code in skill_codes:
            rows.append({
                col_acquired_member: member_code,
                col_competence_type: col_skill_type,
                col_competence_code: skill_code,
                col_competence_name: f'Skill_{skill_code}',
                col_level: np.random.choice([0, 1, 2, 3], 1)[0],
            })

    acquired_df = pd.DataFrame(rows)

    # Create skill dataframe (no special column names needed based on error)
    n_cat_a = n_skills // 2
    n_cat_b = n_skills - n_cat_a  # Ensure total is n_skills
    skill_df = pd.DataFrame({
        'スキルコード': skill_codes,
        'スキル名': [f'Skill_Name_{code}' for code in skill_codes],
        'カテゴリー': ['Category_A'] * n_cat_a + ['Category_B'] * n_cat_b,
    })

    # Create education and license dataframes
    education_df = pd.DataFrame({
        'メンバーコード': member_codes[:n_members // 2],
        '教育コース': ['Course_1'] * (n_members // 2),
    })

    license_df = pd.DataFrame({
        'メンバーコード': member_codes[:n_members // 3],
        '資格名': ['License_1'] * (n_members // 3),
    })

    return member_df, acquired_df, skill_df, education_df, license_df

def test_layer_1_3_architecture():
    """Test the complete Layer 1-3 architecture"""

    logger.info("=" * 80)
    logger.info("Testing Layer 1-3 Causal Skill Profiling Architecture")
    logger.info("=" * 80)

    # 1. Create mock data
    logger.info("\n[1/4] Creating mock data...")
    member_df, acquired_df, skill_df, education_df, license_df = create_mock_data(n_members=50, n_excellent=5)
    logger.info(f"✓ Created mock data: {len(member_df)} members, {len(skill_df)} skills")

    # 2. Initialize analyzer and load data
    logger.info("\n[2/4] Initializing TalentAnalyzer...")
    analyzer = TalentAnalyzer()
    analyzer.load_data(member_df, acquired_df, skill_df, education_df, license_df)
    logger.info(f"✓ Analyzer initialized and data loaded")

    # 3. Train model with excellent members
    logger.info("\n[3/4] Training GNN model...")
    col_member_code = get_config('column_names.member.code', 'メンバーコード  ###[member_code]###')
    excellent_members = member_df[col_member_code].head(5).tolist()
    logger.info(f"Excellent members: {excellent_members}")

    try:
        analyzer.train(excellent_members, epochs_unsupervised=10)
        logger.info("✓ Model training completed")
    except Exception as e:
        logger.error(f"✗ Model training failed: {e}")
        return False

    # 4. Run Layer 1-3 analysis
    logger.info("\n[4/4] Running Layer 1-3 analysis...")

    try:
        # Layer 1: Analyze skill profile of excellent members
        logger.info("  → Running Layer 1: Skill profile analysis...")
        skill_profile = analyzer.analyze_skill_profile_of_excellent_members(excellent_members)
        logger.info(f"✓ Layer 1 completed: Found {len(skill_profile)} skills")

        if len(skill_profile) == 0:
            logger.warning("  ⚠ Warning: skill_profile is empty")
        else:
            top_skill = skill_profile[0]
            logger.info(f"  → Top skill: {top_skill['skill_name']} (importance: {top_skill['importance']:.3f})")

        # Layer 2: Estimate heterogeneous treatment effects
        logger.info("  → Running Layer 2: HTE estimation...")
        hte_results = analyzer.estimate_heterogeneous_treatment_effects(excellent_members, skill_profile)
        logger.info(f"✓ Layer 2 completed: Got HTE results for {len(hte_results)} members")

        # Layer 3: Generate comprehensive insights
        logger.info("  → Running Layer 3: Insight generation...")
        insights = analyzer.generate_comprehensive_insights(excellent_members, skill_profile, hte_results)
        logger.info(f"✓ Layer 3 completed: Generated {len(insights)} types of insights")

        # Verify insights structure
        expected_keys = ['executive_summary', 'top_10_skills', 'organizational_gaps',
                        'priority_recommendations', 'skill_synergies', 'development_roadmap']
        missing_keys = [k for k in expected_keys if k not in insights]

        if missing_keys:
            logger.warning(f"  ⚠ Missing keys in insights: {missing_keys}")
        else:
            logger.info(f"  ✓ All expected insight keys present")

        logger.info("\n" + "=" * 80)
        logger.info("✅ Layer 1-3 Architecture Test PASSED")
        logger.info("=" * 80)

        return True

    except Exception as e:
        logger.error(f"\n✗ Layer 1-3 analysis failed: {e}", exc_info=True)
        logger.error("\n" + "=" * 80)
        logger.error("❌ Layer 1-3 Architecture Test FAILED")
        logger.error("=" * 80)
        return False

if __name__ == "__main__":
    success = test_layer_1_3_architecture()
    exit(0 if success else 1)
