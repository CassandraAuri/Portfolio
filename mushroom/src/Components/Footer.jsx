import React from 'react'
import { NavBarList } from '../Components/NavbarList'
//
import discord from '../Components/Photos/discordlogo.png'
import instagram from '../Components/Photos/instagram.png'
import '../Components/footer.css'
import SignUp from './SignUp'
const Footer = () => {
    return ( // Navigation
        <div className="footer">
            <div>
            <SignUp />
            </div>
            <div>
                <h1>Navigation</h1>
            <ul className="navfooter">
                {NavBarList.map((items) => {
                    return (
                        <li className="nav-links">
                            <a href={items.url} className="nav-links" >
                                {items.name}
                            </a>
                        </li>
                    )
                })}

            </ul>
            </div>
            <div>
            <h1> Social Media
                <ul className="social">
                    <li><a href="https://discord.gg/KZj5C7wVDs">
                        <img src={discord} width="50" height="50" />
                    </a>
                    </li>
                    <li><a href="https://www.instagram.com/ualbertamushrooms/?hl=en">
                        <img src={instagram} width="50" height="50" />
                    </a>
                    </li>
                </ul>

            </h1>
            </div>
        </div> //Social media

    )
};


export default Footer
