import { model, Schema } from "mongoose";

const infoSchema = new Schema({
  name: String,
gender: String,
age: Number,
height: Number,
weight: Number,
  // seat: String
})

const Info = model("info", infoSchema);
export default Info;

