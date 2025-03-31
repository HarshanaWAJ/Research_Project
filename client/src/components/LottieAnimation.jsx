// src/components/LottieAnimation.js
import React from 'react';
import Lottie from 'react-lottie';

const LottieAnimation = ({ animationData }) => {
  const options = {
    loop: true, // Set to true to loop the animation indefinitely
    autoplay: true, // Set to true to start animation automatically
    animationData: animationData,
    rendererSettings: {
      preserveAspectRatio: 'xMidYMid slice',
    },
  };

  return (
    <div className="lottie-container" style={{ width: '100px', height: '100px', alignItems: 'center', alignContent: 'center'}}>
      <Lottie options={options} />
    </div>
  );
};

export default LottieAnimation;
