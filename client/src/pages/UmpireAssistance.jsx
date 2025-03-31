import React, { useState, useEffect } from 'react';
import '../styles/UmpireAssistance.css';

// Import Images (if needed, not used in this case)
import image1 from '../assets/images/image1.jpg';
import image2 from '../assets/images/image2.png';
import image3 from '../assets/images/image3.png';
import image4 from '../assets/images/image4.png';

// Import Animations
import animationData1 from '../assets/animations/out.json';
import animationData2 from '../assets/animations/animation2.json';
import animationData4 from '../assets/animations/wide.json';

// Import Animation Rendering JSX
import LottieAnimation from '../components/LottieAnimation';  // Import the Lottie component

// Importing Link
import { Link } from 'react-router-dom';

// Constant Variable for Slide Show - Image Container
const images = [image1, image2, image3, image4];

function UmpireAssistance() {
  // Slideshow logic
  const [currentImageIndex, setCurrentImageIndex] = useState(0);
  const [fade, setFade] = useState(true);  // Start with fade-in

  useEffect(() => {
    const interval = setInterval(() => {
      setFade(false);  // Trigger fade-out
      setTimeout(() => {
        setCurrentImageIndex((prevIndex) => (prevIndex + 1) % images.length);  // Change image
        setFade(true);  // Trigger fade-in
      }, 500);  // Short time for fade-out (500ms)
    }, 5500);  // Adjust interval time slightly more than the fade-out time (5500ms)
    
    return () => clearInterval(interval); // Clean up interval on component unmount
  }, []);

  // Card data including animation for each card - Card Container
  const cardData = [
    {
      id: 1,
      title: 'LBW Classification',
      description: 'Some quick example text for LBW Classification.',
      animation: animationData1, // Lottie animation for this card
      url: '/lbw-clasification',
      btn_text: 'LBW Classification', // Button text for this card
    },
    {
      id: 2,
      title: 'Wide Ball Classification',
      description: 'Some quick example text for Wide Ball Classification.',
      animation: animationData4, // Lottie animation for this card
      url: '/wide-clasification',
      btn_text: 'Wide Ball Classification', // Button text for this card
    },
    {
      id: 3,
      title: 'No Ball Classification',
      description: 'Some quick example text for No Ball Classification.',
      animation: animationData2, // Lottie animation for this card
      url: '/noball-clasification',
      btn_text: 'No Ball Classification', // Button text for this card
    },
  ];

  return (
    <div className="umpire-assistance-main">
      <div className="left-section">
        <div className="content-wrapper">
          <div className="heading-one">Cricket Vision</div>
          <div className="heading-two">Umpire Assistant System</div>
        </div>
        {/* Slideshow container */}
        <div
          className={`slideshow ${fade ? 'fade-in' : ''}`}
          style={{
            backgroundImage: `url(${images[currentImageIndex]})`,
          }}
        />
      </div>

      <div className="right-section">
        <div className="heading">Features</div>

        {/* Render the cards dynamically from cardData */}
        {cardData.map((card) => (
          <div className="card-view m-1 p-1" key={card.id}>
            <div className="card">
              <div className="card-img-top">
                {/* Render Lottie animation for each card */}
                <LottieAnimation animationData={card.animation} />
              </div>
              <div className="card-body">
                <h5 className="card-title">{card.title}</h5>
                <p className="card-text">{card.description}</p>
                <Link to={card.url} className="btn btn-primary">
                  {card.btn_text} 
                </Link>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

export default UmpireAssistance;
