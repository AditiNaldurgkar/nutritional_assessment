import express from "express";

import Reservation from './models/Reservation.js';
const app = express();
const PORT = process.env.PORT || 5000;
import cors from "cors";
import mongoose, { connect } from "mongoose";
import dotenv from "dotenv";
dotenv.config();

app.use(cors());
app.use(express.json());

const connectDB = async () => {
  mongoose.connect(process.env.MONGODB_URL);
  console.log("Database Connected");
};

connectDB();

app.get("/health", (req, res) => {
  res.json({
    success: true,
    message: "Server is Running",
    data: null,
  });
});

// ==============   All APIs here  =============
// Review Api
import Review from "./models/Review.js";

app.post("/review", async (req, res) => {
  const { name, message, userPhoto } = req.body;

  if (!name) {
    return res.json({
      success: false,
      message: "Login is required",
      data: null,
    });
  }

  if (!message) {
    return res.json({
      success: false,
      message: "Review is required",
      data: null,
    });
  }

  const newReview = await Review.create({
    name: name,
    message: message,
    userPhoto: userPhoto
  });

  res.json({
    success: true,
    message: "Review added successfully",
    data: newReview,
  });
});

app.get("/review", async (req, res) => {
  const review = await Review.find();

  res.json({
    success: true,
    message: "Review featched successfully",
    data: review,
  });
});
app.delete("/review/:id", async (req, res) => {
  const { id } = req.params;

  await Review.deleteOne({ _id: id });

  res.json({
    success: true,
    message: "Review deleted successfully",
    data: null,
  });
});

// New GET endpoint for the root path
app.get("/", (req, res) => {
  res.json({
    success: true,
    message: "Welcome to the Review page",
    data: null,
  });
});

// Login Api

import User from "./models/User.js"
app.post("/user", async (req, res) => {
  const { email, userName, userPhoto } = req.body;

  try {

    let existingUser = await User.findOne({ email });

    if (existingUser) {

      existingUser.isLoggedIn = true;
      await existingUser.save();

      res.json({
        success: true,
        message: "User logged in successfully",
        data: existingUser
      });
    } else {

      const newUser = await User.create({
        email,
        userName,
        userPhoto,
        isLoggedIn: true
      });

      res.json({
        success: true,
        message: "New user created and logged in successfully",
        data: newUser
      });
    }
  } catch (error) {
    console.error("Error creating or updating user:", error);
    res.status(500).json({
      success: false,
      message: "Internal server error",
      data: null
    });
  }
});


app.get("/user", async (req, res) => {
  try {
    const users = await User.find();
    res.json({
      success: true,
      message: "Users fetched successfully",
      data: users
    });
  } catch (error) {
    console.error("Error fetching users:", error);
    res.status(500).json({
      success: false,
      message: "Internal server error",
      data: null
    });
  }
});


// Logout Api
app.post("/user/logout", async (req, res) => {
  const { email } = req.body;

  try {

    let user = await User.findOne({ email });
    if (user) {
      user.isLoggedIn = false;
      await user.save();
      res.json({
        success: true,
        message: "User logged out successfully",
        data: user
      });
    } else {
      res.status(404).json({
        success: false,
        message: "User not found",
        data: null
      });
    }
  } catch (error) {
    console.error("Error logging out user:", error);
    res.status(500).json({
      success: false,
      message: "Internal server error",
      data: null
    });
  }
});

// Payment Api
import Payment from "./models/paymentpage_model.js";
import Contact from "./models/ContactUs.js";

app.post("/payment", async (req, res) => {
  const {
    First_Name,
    Last_Name,
    Date_of_Birth,
    Phone_Number,
    Email,
    City,
    Zip
  } = req.body;

  try {
    const newPayment = await Payment.create({
      First_Name: First_Name,
      Last_Name: Last_Name,
      Date_of_Birth: Date_of_Birth,
      Phone_Number: Phone_Number,
      Email: Email,
      City: City,
      Zip: Zip
    });

    res.status(201).json({
      message: "Payment created successfully",
      data: newPayment
    });
  } catch (error) {
    console.error("Error creating payment:", error);
    res.status(500).json({ error: "Internal server error" });
  }
});


//Reservation Api
app.post("/reservation", async (req, res) => {
  const { name, phone, gender, seat, email } = req.body;

  try {
    const book = await Reservation.create({ name, phone, email, gender, age });
    res.json({
      success: true,
      message: "Details added Successfully",
      data: book
    });
  } catch (error) {
    console.error('Error creating reservation:', error);
    res.status(500).json({
      success: false,
      message: "Error adding reservation",
      error: error.message
    });
  }
});

app.get("/reservation", async (req, res) => {
  try {
    const book = await Reservation.find();
    res.json({
      success: true,
      message: "Reservation Details fetched successfully",
      data: book
    });
  } catch (error) {
    console.error('Error fetching reservation:', error);
    res.status(500).json({
      success: false,
      message: "Error fetching Reservation",
      error: error.message
    });
  }
});

// Contact Us Api
app.post("/contact", async (req, res) => {
  const { name, email, phone, message } = req.body;
  const contact = await Contact.create({ name, email, phone, message });
  res.json({
    success: true,
    message: "Your Message has been sent to the admin",
    data: contact
  });
});

app.get("/contact", async (req, res) => {
  const contacts = await Contact.find();
  res.json({
    success: true,
    message: "Contact Requests fetched successfully",
    data: contacts
  });

});

// Signup API
// Signup API
app.post("/signup", async (req, res) => {
  const { name, email, password } = req.body;

  // Validate input
  if (!name || !email || !password) {
    return res.status(400).json({
      success: false,
      message: "All fields are required",
      data: null,
    });
  }

  try {
    // Check if the user already exists
    const existingUser = await User.findOne({ email });

    if (existingUser) {
      return res.status(400).json({
        success: false,
        message: "User already exists",
        data: null,
      });
    }

    // Hash the password
    const bcrypt = await import("bcrypt");
    const salt = await bcrypt.genSalt(10);
    const hashedPassword = await bcrypt.hash(password, salt);

    // Create the user
    const newUser = await User.create({
      name,
      email,
      password: hashedPassword,
    });

    res.status(201).json({
      success: true,
      message: "User registered successfully",
      data: newUser,
    });
  } catch (error) {
    console.error("Error during signup:", error);
    res.status(500).json({
      success: false,
      message: "Internal server error",
      data: null,
    });
  }
});

// Admin Api
import Admin from "./models/Admin.js";

app.post("/admin", async (req, res) => {
  const { email, password } = req.body;

  const admin = await Admin.create({ email, password });

  res.json({
    success: true,
    message: "Your Message has been sent to the admin",
    data: admin
  })

});

app.get("/admin", async (req, res) => {
  try {
    const admin = await Admin.findOne(); // Retrieve the first admin from the database
    if (admin) {
      res.json({
        success: true,
        message: "Admin details fetched successfully",
        data: admin
      });
    } else {
      res.status(404).json({
        success: false,
        message: "Admin details not found"
      });
    }
  } catch (error) {
    console.error('Error fetching admin details:', error);
    res.status(500).json({
      success: false,
      message: "Failed to fetch admin details"
    });
  }
});
const handleImageUpload = async (event) => {
  const file = event.target.files[0];
  const formData = new FormData();
  formData.append("image", file);

  const res = await fetch("http://localhost:5000/analyze", {
    method: "POST",
    body: formData,
  });

  const data = await res.json();
  alert("Prediction: " + data.status);
};

<input type="file" onChange={handleImageUpload} />


app.listen(PORT, () => console.log(`Server running on port ${PORT}`));
