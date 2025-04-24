import { model , Schema} from 'mongoose';

const paymentSchema = new Schema({
    First_Name: String,
    Last_Name: String,
    Date_of_Birth: Number,
    Phone_Number: Number,
    Email: String,
    City: String,
    Zip: Number
})

const Payment = model("Payment" , paymentSchema);

export default Payment;