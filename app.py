from flask import Flask, request,render_template
import pickle
import pandas as pd
#from Normalization import normalize_num_data

app = Flask(__name__)
model = pickle.load(open('gboost_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template("home1.html")

@app.route('/prediction',methods=['POST'])
def prediction():
    if(request.method == "POST"):
         # vehicle_age = int(request.form.get("Vehicle_Age"))
         # gender = int(request.form.get("Gender"))
         # driving_licence = int(request.form.get("Driving_Licence"))
         # previously_insured = int(request.form.get("Previously_Insured"))
         # vehicle_damage = int(request.form.get("Vehicle_Damage"))
         # age = int(request.form.get("Age"))
         # annual_premium = int(request.form.get("Annual_Premium"))
         # print(vehicle_age,gender,driving_licence,previously_insured,vehicle_damage,age,annual_premium)
         # print(request.form.lists())
         data = [request.form.values()]
        # data = [[vehicle_age,gender,driving_licence,previously_insured,vehicle_damage,age,annual_premium]]
#         data = [int(x) for x in request.form.values()]
         final_data = pd.DataFrame(data,columns=['Vehicle_Age','Gender','Driving_Licence','Previously_Insured','Vehicle_Damage','Age','Annual_Premium'])
         predi = model.predict(final_data)
         print(predi)
        #  print(final_data)
        #  num_data = final_data[['Age','Annual_Premium']]
        #  final = normalize_num_data(num_data)
        #  print(final)
        #  final_data['Age','Annual_premium'] = final['Age','Annual_Premium']
        #  print(final_data)   
            
           
            
    return render_template("home1.html",prediction_text ="Prediction : {}".format("Yes\n Customer might buy Insurance" if predi==1 else "No\n Customer might not buy Insurance"))
        

if __name__ == "__main__":
    app.run()