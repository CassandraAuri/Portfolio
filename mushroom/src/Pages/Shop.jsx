import React from 'react'
import Cart from '../Components/ShopRelated/Cart'
import {useState} from 'react'
import ShopItem from '../Components/ShopRelated/Purchase'
export const Context= React.createContext([]);
const Shop =()=>{
  const [price,setPrice]=useState()
  const [cart,setCart]=useState([])
  return(
    <div>
<Cart cart={cart} setCart={setCart} price={price} setPrice={setPrice} />
<Context.Provider value={[cart,setCart]}>
<ShopItem cart={cart} setCart={setCart} context={Context}/></Context.Provider>
  </div>
  )
};
export default Shop