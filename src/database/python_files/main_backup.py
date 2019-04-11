# # Load Dataset

def get_all_patients_result():

    subj_range = np.hstack((np.arange(2001,2002),np.arange(3001,3006)))
    all_patients = [str(i) for i in subj_range]

    # In[37]:

    cleaned_data_path = datapath + 'cleaned/cleaned_data_' + all_patients[0] + '_to_' + all_patients[-1] + '.csv'
    predicted_data_path = datapath + 'prediction/predicted_data_' + all_patients[0] + '_to_' + all_patients[-1] + '.csv'
    all_day_summary_path = datapath + 'summary/all_day_summary_' + all_patients[0] + '_to_' + all_patients[-1] + '.csv'
    act_period_path = datapath + 'summary/activity_period_' + all_patients[0] + '_to_' + all_patients[-1] + '.csv'


    df_all_p_sorted = load_all_data(all_patients)
    export_cleaned_data(df_all_p_sorted, cleaned_data_path)

    # # Predict Labels

    df_all_p_sorted = predict_label(predicted_data_path)
    export_predicted_data(df_all_p_sorted, predicted_data_path)

    # In[39]:

    df_summary_all, df_act_period = get_summarized_data(predicted_data_path)
    export_summarized_data(df_summary_all, df_act_period, all_day_summary_path, act_period_path)

    return df_all_p_sorted, df_summary_all, df_act_period