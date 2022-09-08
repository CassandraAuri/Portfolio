import Navbar from '../Components/Navbar'
import logo from '../Components/Photos/logo.png'
import Signuptiles from '../Components/Tiles-home'
import Mailinglist from '../Components/Mailinglist'
import Projects from '../Components/Projects'
import SignUp from '../Components/SignUp'
import '../Components/home.css'
function Home() {
    return (
        <div ClassName="total-page">
            <div className="hero-display">
                <div className="hero">

                    <img src={logo} height="350" width="350" className="hero" />
                </div>
                <div className="right">
                    <p className="hero-text">
                        The UofA Mycology Club is a place for mycophiles to share their mycological passion,
                        whether it be  in the realm of cultivation, art, culinary, or research.
                        Being a member of the UofA Mycology Club is a great way to learn about fungus,
                        network with other mycoinnovative people, and contribute back to the community
                        in a fun way! To join this club you do not need to be a mushroom master,
                        casual appreciation of mushrooms is always welcome.
                        Below are the buttons to register
                    </p>
                </div>
            </div>
            <div>
                <Signuptiles></Signuptiles>
            </div>
            <div>
                <Projects></Projects>
            </div>
            <br/>
            <br/>
        </div>


    )
} export default Home