import mongoose from "mongoose";
import  {model,Schema} from  mongoose;
const PatientSchema = new Schema({
name : String ,
age :Number,
gender : String,
height:Number,
weight:Number,
BMI: Number,
});

const Patient= model('Patient',PatientSchema);
export default Patient;