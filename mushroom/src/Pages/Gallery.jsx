import React, {useState} from 'react';
import {SliderData} from '../Components/SliderData';
import leftarrow from '../Components/Photos/leftarrow.png'
import rightarrow from '../Components/Photos/rightarrow.png'

const Gallery = ({slides}) =>{
    const [current,setCurrent] = useState()
    return(
        <section className="Slider">
            <img src={leftarrow}className="left-arrow"/>
            <img src={rightarrow} className="right-arrow"/>
            {SliderData.map((slider,index) =>{
                return <img src={slider.image} alt="gallery photo"/>
            }
            )}
        </section>
    )
}
export default Gallery;