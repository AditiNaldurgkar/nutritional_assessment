import { createBrowserRouter, RouterProvider } from "react-router-dom";
import React from 'react';
import './App.css';
import Home from "./views/Home/Home";
import Reservation from "./views/Reservation/Reservation";
import Login from "./views/Login/Login";
import Review from "./views/Review/Review";
import Dashboard from "./views/Dashboard/Dashboard";
import Contact from "./views/Contact/Contact";
import Payment from "./components/Payment/Payment";
import Admin from "./views/Admin/Admin";
import Chatbot from "./views/Chatbot/Chatbot"
import Signup from "./views/Signup/Signup";
import Info from "./views/Info/Info";
import Checker from "./views/Checker/Checker";
// import AdminLogin from "./views/Admin/AdminLogin"

function App() {
  const router = createBrowserRouter([
    {
      path: "/",
      element: <Home/>
    },
    {
      path: "/contact",
      element: <Contact/>
    },
    {
      path: "/review",
      element: <Review/>
    },

    {
      path: "/reservation",
      element: <Reservation/>
    },
    
    {
      path: "/chatbot",
      element: <Chatbot/>
    },
    {
      path: "/payment",
      element: <Payment/>
    },
    {
      path: "/login",
      element: <Login/>
    },
    {
      path:"/signup",
      element:<Signup/>
    },
    {
      path: "/admin",
      element : <Admin/>
    },
    {
      path: "/dashboard",
      element: <Dashboard/>
    },
    {
      path: "/info",
      element: <Info/>
    },
    {
      path: "/checker",
      element: <Checker/>
    }
  ])

  return (
    <>
    {/* === ROUTING HERE  === */}
    <RouterProvider router={router}/>
    
    </>
  );
}
export default App;
