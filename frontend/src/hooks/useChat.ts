import { useCallback, useEffect, useRef, useState } from "react";
import {
  getClothingAppHealth,
  getHealth,
  chat as postChat,
  getProductDetails,
  getCart,
  updateCartQuantity,
  removeCartItem,
} from "../api/agent";
import { ApiError } from "../api/http";
import type { CartView, HealthStatus, TimelineMessage, ProductView } from "../types";

const INITIAL_HEALTH: HealthStatus = {
  agent: "checking",
  clothingApp: "checking",
  llm: "unknown",
};

function makeLocalMessage(role: "user" | "assistant", content: string): TimelineMessage {
  return {
    id: crypto.randomUUID(),
    role,
    content,
    createdAt: new Date().toISOString(),
  };
}

export function useChat() {
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [messages, setMessages] = useState<TimelineMessage[]>([]);
  const [cart, setCart] = useState<CartView | null>(null);
  const [activeDetailsProduct, setActiveDetailsProduct] = useState<ProductView | null>(null);
  const [isSending, setIsSending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [health, setHealth] = useState<HealthStatus>(INITIAL_HEALTH);
  const healthStarted = useRef(false);

  const checkHealth = useCallback(async () => {
    const next: HealthStatus = {
      agent: "offline",
      clothingApp: "offline",
      llm: "unknown",
    };
    try {
      const result = await getHealth();
      next.agent = result.agent.status === "ok" ? "online" : "offline";
      next.clothingApp = result.ready.status === "ready" ? "online" : "offline";
      next.llm = result.ready.llm === "configured" ? "configured" : "local_fallback";
    } catch {
      try {
        const app = await getClothingAppHealth();
        next.clothingApp = app.status === "ready" ? "online" : "offline";
      } catch {
        next.clothingApp = "offline";
      }
    }
    setHealth(next);
  }, []);

  useEffect(() => {
    if (healthStarted.current) return;
    healthStarted.current = true;
    void checkHealth();
  }, [checkHealth]);

  const newConversation = useCallback(() => {
    setConversationId(null);
    setMessages([]);
    setCart(null);
    setError(null);
  }, []);

  const sendMessage = useCallback(async (rawMessage: string) => {
    const message = rawMessage.trim();
    if (!message || isSending) return;

    setError(null);
    setMessages((current) => [...current, makeLocalMessage("user", message)]);
    setIsSending(true);

    try {
      const sessionId = conversationId || crypto.randomUUID();
      if (!conversationId) {
        setConversationId(sessionId);
      }
      
      const response = await postChat(message, sessionId);
      
      let fullProducts: (ProductOption | ProductView)[] = [];
      if (response.products && response.products.length > 0) {
        fullProducts = response.products;
      } else if (response.state?.product_cards && response.state.product_cards.length > 0) {
        fullProducts = response.state.product_cards.map(c => c.product);
      }
      
      if (response.state?.current_intent === "get_details" && fullProducts.length === 1 && "product_name" in fullProducts[0]) {
        setActiveDetailsProduct(fullProducts[0] as unknown as ProductView);
      }

      if (response.cart) {
        setCart(response.cart as unknown as CartView);
      } else if (response.state?.cart_card) {
        setCart({
          ...response.state.cart_card,
          total_quantity: response.state.cart_card.item_count,
        } as unknown as CartView);
      } else if (!response.state?.cart?.cart_id) {
        setCart(null);
      }

      let replyContent = response.reply;
      let checkoutPreview = response.checkout || response.state?.checkout_card || null;

      let deliveryContext = undefined;
      if (checkoutPreview) {
        deliveryContext = response.state?.delivery;
      }

      setMessages((current) => [
        ...current,
        {
          id: crypto.randomUUID(),
          role: "assistant",
          content: replyContent,
          createdAt: new Date().toISOString(),
          products: fullProducts.length > 0 ? fullProducts : undefined,
          product: response.product,
          checkout: response.checkout,
          order: response.order,
          contentType: response.content_type,
          uiActions: response.ui_actions,
          checkoutPreview,
          deliveryContext,
        },
      ]);
    } catch (reason) {
      const messageText = reason instanceof ApiError
        ? reason.message
        : "The request failed. Please try again.";
      setError(messageText);
      setMessages((current) => [
        ...current,
        makeLocalMessage("assistant", `I hit a connection problem: ${messageText}`),
      ]);
      void checkHealth();
    } finally {
      setIsSending(false);
    }
  }, [checkHealth, conversationId, isSending]);

  const updateCartItemQuantity = useCallback(async (itemId: string, quantity: number) => {
    if (!cart?.cart_id) return;
    try {
      const updatedCart = await updateCartQuantity(cart.cart_id, itemId, quantity);
      setCart(updatedCart);
    } catch (e) {
      console.warn("Failed to update cart quantity", e);
    }
  }, [cart]);

  const removeCartItemFromCart = useCallback(async (itemId: string) => {
    if (!cart?.cart_id) return;

    try {
      const updatedCart = await removeCartItem(cart.cart_id, itemId);
      setCart(updatedCart);
    } catch (e) {
      console.warn("Failed to remove item", e);
    }
  }, [cart]);

  return {
    conversationId,
    messages,
    cart,
    suggestedActions: [],
    isStarting: false,
    isSending,
    error,
    health,
    activeDetailsProduct,
    closeDetails: () => setActiveDetailsProduct(null),
    sendMessage,
    newConversation,
    checkHealth,
    updateCartItemQuantity,
    removeCartItemFromCart,
  };
}
