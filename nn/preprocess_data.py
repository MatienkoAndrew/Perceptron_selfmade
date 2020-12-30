from .min_max_scaler import MinMaxScaler

def preprocess(df):
	df.drop(['1'], axis=1, inplace=True)
	df['target'] = df['target'].apply(lambda x: 1 if x == 'M' else 0)
	not_normalized_cols = ['3', '4', '5', '6', '15', '16', '23', '24', '25', '26']
	scaler = MinMaxScaler()
	df[not_normalized_cols] = scaler.fit_transform(df[not_normalized_cols])
	target = df['target'].copy()
	df.drop(['target'], axis=1, inplace=True)

	X, y = df, target
	return X, y