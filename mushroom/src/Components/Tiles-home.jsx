// tiles: Bearsdens sign up, discord, mailing list. 
import Bearsdens from '../Components/Photos/bearsden.png'
import discord from '../Components/Photos/discord.png'
import mail from '../Components/Photos/mail.png'
import './tiles.css'
const Signuptiles = () => {
    return (
        <div className="AboutTiles">
            <h1 className="header">How to join!</h1>
            <ul className="cards">
                <li className="card1">
                    <img src={Bearsdens} height="50px" width="150px" />
                    <h2>
                        Bearsden
                    </h2>
                    <br />
                    <p>
                        To sign up officially for the club, find us on Bearsden using the link below
                    </p>
                    <footer>
                        <form action="https://alberta.campuslabs.ca/engage/organization/ualbertamushrooms">
                            <input type="submit" value="Learn more" className="button1" />
                        </form>
                    </footer>
                </li>
                <li className="card1">
                    <img src={discord} height="50px" width="150px" />
                    <h2>
                        Discord
                    </h2>
                    <br />
                    <p>
                        Our main source of communication, Sign up for discord and follow the link below to get involved instantly!
                    </p>
                    <footer>
                        <form action="https://discord.gg/KZj5C7wVDs" >
                            <input type="submit" value="Learn more" className="button2" />
                        </form>
                    </footer>
                </li>

            </ul>
        </div>



    )
};
export default Signuptiles