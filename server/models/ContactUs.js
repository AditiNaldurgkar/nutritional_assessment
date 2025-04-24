import { model, Schema } from "mongoose";

const ContactSchema = new Schema({
    name: String,
    email: String,
    phone : String,
    message : String
})

const Contact =  model("Contact", ContactSchema);
export default Contact;

