import Item from './ShopItem'
import {items} from './shoplist'
const ShopItem =({cart,setCart,setQuantity, quantity})=>{
    const addCart = (item) => (qty) =>{
        item.qty = qty
        setCart(cart=>[...cart,item])
    }
    return(
        <div>
            <h1>Shop</h1>
                <ul className="list">
                {items.map((item, index)=>{
                                return(
                                    <Item addCart={addCart(item)} item={item} />
                                )})}
                            </ul>
        </div>
    )}       
export default ShopItem