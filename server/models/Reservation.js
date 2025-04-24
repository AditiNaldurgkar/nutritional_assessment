import { model, Schema } from "mongoose";

const reservationScheme = new Schema({
  name: String,
  phone: String,
  email :String,
gender: String,
age: Number,
  // seat: String
})

const Reservation = model("reservation", reservationScheme);
export default Reservation;

