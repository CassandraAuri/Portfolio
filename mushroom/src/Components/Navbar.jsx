import React, {Component} from 'react';
import logo from '../Components/Photos/logo.png'
import bars from '../Components/Photos/bars.png'
import './navbar.css';
import { NavBarList } from './NavbarList';
 class Navbar extends Component {
    state = {clicked: false}
    handleClick = () =>{
        this.setState({clicked: !this.state.clicked})
    }
    render(){
    return(
        <nav className ="NavbarItems">
            <h1><img src={logo} width = "100" height = "100" className='logo'/></h1>
            
        <ul className={this.state.clicked ? 'nav-menu-active': 'nav-menu'} >
         {NavBarList.map((items)=>{
            return(
                <li >
                    <a href={items.url} className="nav-link" >
                        {items.name}
                    </a>
                </li>
            ) })}
        </ul>
<div><img onClick={this.handleClick} src={bars} width = "100" height = "100"  className='navigator'/></div>
        </nav>
    )
  }
}
export default Navbar

