import React, { useState } from "react";

export default FlipCard = (props) => {
  return (
    <div className="w-full flex justify-center items-center p-4">
      <div className="group [perspective:1000px]">
        <div className="relative w-80 h-96 transition-transform duration-700 [transform-style:preserve-3d] group-hover:[transform:rotateY(180deg)]">
          {/* Front Side */}
          <div className="absolute inset-0 bg-blue-500 text-white rounded-2xl shadow-xl flex flex-col justify-center items-center [backface-visibility:hidden]">
            <div className="text-5xl mb-4">🖱️</div>
            <h2 className="text-2xl font-bold">{props.frontTitle}</h2>
            <p className="mt-2 text-center px-6">{props.frontDesc}</p>
          </div>

          {/* Back Side */}
          <div className="absolute inset-0 bg-blue-600 text-white rounded-2xl shadow-xl flex flex-col justify-center items-center [transform:rotateY(180deg)] [backface-visibility:hidden]">
            <h2 className="text-2xl font-bold">{props.backTitle}</h2>
            <p className="mt-2 text-center px-6">{props.backDesc}</p>
            <button className="mt-6 px-4 py-2 bg-white text-blue-600 font-semibold rounded-xl shadow hover:bg-gray-100 transition">
            {props.backButton}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
