import argparse
from tokenize import String
from Functions import *

parser = argparse.ArgumentParser(description="Train model")
parser.add_argument("--country_test", type=str, help="Country to test model on", required=False)
parser.add_argument("--no_days", type=int, help="Number of days to predict", required=False)
parser.add_argument("--Folder", type=str, help="Folder with training on one or two", required=False)
parser.add_argument("--train_test_predict", type=str, help="Type: train (train model), test (test model), predict (predict future)", required=False)
parser.add_argument("--save_directory", type=str, help="name of model to save", required=False)
parser.add_argument("--load_model", type=str, help="name of model to load", required=False)
args = parser.parse_args()

    
def main(test_train_pred, country_test, nr_days, load_model_choice, chosen_directory, folder):
    """This is a main function which gives you 3 options, train models, test models, or use a model to predict future.

    Args:
        test_train_pred (string): train/test/predict depending on what you want to do.
        country_test (string): Country to test your model on.
        nr_days (int): Number of future days you want to predict. 
        load_model_choice (string): Name of the model you want to use.
        chosen_directory (string): Name of the directory you want to save your trained models in. 
        folder (string): Name of the folder where your model of choice is. 
    """

    method = st.sidebar.selectbox("Select method", ("Start", "Predict future days", "Test model"))
    load_model_choice = st.sidebar.selectbox("Select model", ("David Lacosse","Marie Delgado"))
    if method == "Start":
        st.header("This is your Covid prediction app")
        st.subheader("In this app you can choose between two different models")
        st.write("Please select an optional method in the sidebar to the left, then select a model")
    elif method == "Test model":
        st.header("In this method you can test the model on Norways confirmed cases")
        test_train_pred = "test"
    elif method == "Predict future days":
        st.header("In this method you can use the chosen model to forecast coming days")
        test_train_pred = "predict"
    else: pass
        
    folder = "Streamlit modeller"
    country_test = "Norway"
    
    df = pd.read_csv("CONVENIENT_global_confirmed_cases.csv")

    if test_train_pred == "train": #--train_test_predict (train)
        train(df, chosen_directory)

    elif test_train_pred == "test": # python ml.py --train_test_predict "test" --Folder "Jag testar en" --load_model "Nell Garrard" --country_test "Sweden"
        use_model, scaler, dictionary_, rmse_test = load_model_etc(load_model_choice, folder)
        test_pred, rmse, real_data, df_date, df_case = test_new_data(df, country_test, scaler, use_model, dictionary_[6])
        if load_model_choice == "David Lacosse":
            st.write("This model is trained on: Germany, Denmark, Netherlands, Switzerland, Austria")
            real_data = np.pad(real_data, (14,0))
            test_pred = np.pad(test_pred, (14,0))
        elif load_model_choice == "Marie Delgado":
            st.write("This model is trained on: Italy, Portugal, Greece, Ukraine, Spain")
            real_data = np.pad(real_data, (15,0))
            test_pred = np.pad(test_pred, (15,0))
        else: pass
        plot = plots(real_data, test_pred, country_test, df_date, df_case)
        streamlit_test(plot, rmse)
        
    elif test_train_pred == "predict": #Args for predict: python ml.py --train_test_predict "predict" --Folder "Ger_Den_Net_Swi_Aus" --load_model "David Lacosse" --no_days 14 --country_test "Norway" 
        nr_days = st.slider("Select Number of days", 1,60,14)
        use_model, scaler, dictionary_, rmse_test = load_model_etc(load_model_choice, folder)
        X_train_no_scale, y_train_no_scale, X_val_no_scale, y_val_no_scale = preprocess(df, country_test, dictionary_[6])
        y_val_scaled = scaler.transform((y_val_no_scale).reshape(-1,1))
        new_days_pred = predict_future(y_val_scaled, dictionary_[6], use_model, nr_days, scaler)
        streamlit_forecast(new_days_pred)
        

    else: pass

if __name__ == "__main__":

    main(args.train_test_predict, args.country_test, args.no_days, args.load_model, args.save_directory, args.Folder)