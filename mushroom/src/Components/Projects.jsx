import React from 'react'
import sustain from '../Components/Photos/sustain.png'
import outreach from '../Components/Photos/outreach.jpg'
import './projects.css'
const Projects = () => {
    return (
        <div>
            <h1 className="header">
                Projects
            </h1>
            <ul className="cards">
                <li className="card2">
                    <img src={sustain} height="50px" width="150px" />
                    <h2>
                        Sustainability
                    </h2>
                    <br/>
                    <p className="paragraph">Pellentesque habitant morbi tristique senectus et netus et malesuada fames</p>
                </li>
                <li className="card2" >
                    <img src={outreach} height="50px" width="150px" />
                    <h2>
                        Interdiscipinlary outreach
                    </h2>
                    <br />
                    <p>Pellentesque habitant morbi tristique senectus et netus et malesuada fames</p>
<br/>
                </li>

            </ul>
        </div>
    )
};

export default Projects
