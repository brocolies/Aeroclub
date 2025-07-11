def add_route_features(df):
    """searchRoute 기반 feature 추가"""
    df = df.copy()  # 함수 내에서만 copy
    df['is_round_trip'] = df['searchRoute'].str.contains('/', na=False)
    split_routes = df['searchRoute'].str.split('/', expand=True)
    df['home_city'] = split_routes[0]
    df['away_city'] = split_routes[1]
    return df