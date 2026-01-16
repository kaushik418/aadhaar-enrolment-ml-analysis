#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import matplotlib.pyplot as plt
# Load datasets
files = ['api_data_aadhar_enrolment_0_500000.csv',
         'api_data_aadhar_enrolment_500000_1000000.csv',
         'api_data_aadhar_enrolment_1000000_1006029.csv']

df_list = [pd.read_csv(f) for f in files]

aadhaar_df = pd.concat(df_list, ignore_index=True)

# Data cleaning
aadhaar_df.drop_duplicates(inplace=True)
aadhaar_df.fillna(0, inplace=True)

# Date formatting
aadhaar_df['date'] = pd.to_datetime(aadhaar_df['date'])

# Feature engineering
aadhaar_df['child_enrolments'] = aadhaar_df['age_0_5'] + aadhaar_df['age_5_17']

# State‑wise aggregation
state_summary = aadhaar_df.groupby('state')[['age_0_5', 'age_5_17', 'age_18_greater']].sum().reset_index()
# Visualisation
state_summary.set_index('state').plot(kind='bar', stacked=True, figsize=(12,6))
plt.title('State‑wise Aadhaar Enrolment by Age Group')
plt.ylabel('Number of Enrolments')
plt.tight_layout()
plt.show()


# In[4]:


from prophet import Prophet

aadhaar_df['total_enrolments'] = (
    aadhaar_df['age_0_5'] +
    aadhaar_df['age_5_17'] +
    aadhaar_df['age_18_greater']
)

# Prepare data for forecasting
forecast_df = aadhaar_df.groupby('date')['total_enrolments'].sum().reset_index()
forecast_df.columns = ['ds', 'y']

model = Prophet()
model.fit(forecast_df)

future = model.make_future_dataframe(periods=180)
forecast = model.predict(future)

model.plot(forecast)


# In[ ]:




