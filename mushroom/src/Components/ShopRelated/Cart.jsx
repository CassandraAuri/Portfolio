import {Router, Link, Switch} from 'react-router-dom'
var Totalprice=0;
const Cart=({cart,setCart, price, setPrice, quantity})=>{ 
    const removeCart =(item)=>{
        let hardCopy= [...cart]
        hardCopy=hardCopy.filter(cartItem=>cartItem.name!= item.name)
        setCart(hardCopy) /*Removes items from the Cart UseState*/
    }
        setPrice(Totalprice=cart.reduce((Total,Current)=>Total=(Total+Current.qty*Current.price),0))/*Sets price to be displayed to user, NOT CARRIED TO CART*/
        return(
            <div>
                <h1>CART</h1>
                <Link to="/checkout"> {/*Link to checkout, not implemented*/}
                    Proceed to checkout
                </Link>
                <h3>Price:{price}</h3>        
        <ul>
        {cart.map((item, index)=>{
                    return(
                        <li className="indv">
                            <img src={item.image}/>
                            <h2>{item.name}</h2>
                            <h3>{item.price}</h3>
                            <h2>Quantity:{item.qty}</h2> {/*Display*/}
                            <h1>
                            <input type="button" value="Remove from Cart"  onClick={()=>removeCart(item)}/>{/*Calls remove usestate*/}
                            </h1>
                        </li>
                     )})}
            </ul>
    </div>
        )};

export default Cart