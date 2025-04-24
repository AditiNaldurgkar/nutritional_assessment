import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import Button from "../ui/ui";

const Slider = () => {
  const [currentSlide, setCurrentSlide] = useState(0);

  const slides = [
    {
      id: 0,
      title: "Assess Nutritional Health",
      content: "Analyze children's health using images and anthropometric data.",
      image: "https://source.unsplash.com/900x500/?child,health",
    },
    {
      id: 1,
      title: "Track Growth Progress",
      content: "Monitor children's height, weight, and nutritional milestones.",
      image: "https://source.unsplash.com/900x500/?child,growth",
    },
    {
      id: 2,
      title: "Get Personalized Diet Plans",
      content: "Receive AI-powered food recommendations for better nutrition.",
      image: "https://source.unsplash.com/900x500/?healthy,food",
    },
  ];

  const nextSlide = () => {
    setCurrentSlide((prev) => (prev + 1) % slides.length);
  };

  // Auto slide every 5 seconds
  useEffect(() => {
    const interval = setInterval(nextSlide, 5000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="slider-container">
      <AnimatePresence mode="wait">
        <motion.div
          key={slides[currentSlide].id}
          initial={{ opacity: 0, x: 50 }}
          animate={{ opacity: 1, x: 0 }}
          exit={{ opacity: 0, x: -50 }}
          transition={{ duration: 0.8, ease: "easeInOut" }}
          className="slide"
        >
          <img src={slides[currentSlide].image} alt={slides[currentSlide].title} className="slide-image" />
          <div className="slide-content">
            <h2>{slides[currentSlide].title}</h2>
            <p>{slides[currentSlide].content}</p>
            <Button onClick={nextSlide} className="slide-button">
              {currentSlide === slides.length - 1 ? "Restart" : "Next"}
            </Button>
          </div>
        </motion.div>
      </AnimatePresence>
    </div>
  );
};

export default Slider;
