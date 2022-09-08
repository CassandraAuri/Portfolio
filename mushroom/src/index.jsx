import React, {useReducer} from 'react';
import ReactDOM from 'react-dom'
import { BrowserRouter as Router, Route, Switch } from 'react-router-dom';
import About from './Pages/Aboutus'
import NavBar from './Components/Navbar'
import Shop from './Pages/Shop'
import Gallery from './Pages/Gallery'
import Home from './Pages/Home'
import Footer from './Components/Footer'
import Checkout from './Pages/Checkout';
import './index.css';
function App() {
    const reducer = (state, action) => {
        switch (action.type) {
            case 'add-item':
                return { cart: [...state.cart, action.payload] }
        }
    }
    const [state, dispatch] = useReducer(reducer, [])
    return (
        <div className="page">
            <Router>
                <NavBar />
                <Switch>
                    <Route exact path="/" component={Home} className="Route" />
                    <Route exact path="/home" component={Home} className="Route" />
                    <Route path="/about-us" component={About} className="Route">
                    </Route>
                    <Route path="/gallery" component={Gallery} className="Route" />
                    <Route path="/shop" component={Shop} cart={state.cart} dispatch={dispatch} />
                </Switch>
                <Switch>
                    <Route path="/checkout" component={Checkout} cart={state.cart} dispatch={dispatch} />
                </Switch>
                <Footer />
            </Router>
        </div>

    );
}
ReactDOM.render(<App />, document.getElementById('root'))
