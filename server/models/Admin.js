import { model, Schema} from 'mongoose';
const AdminSchema = new Schema({

    email:String,
    password:String
});

const Admin = model('Admin',AdminSchema);
export default Admin;
