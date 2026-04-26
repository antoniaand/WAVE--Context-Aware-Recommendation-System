import api from './api'
import type { RecommendRequest, RecommendResponse, ModelInfo } from '@/types'

export const recommendService = {
  async getRecommendations(payload: RecommendRequest): Promise<RecommendResponse> {
    const { data } = await api.post<RecommendResponse>('/recommend', payload)
    return data
  },

  async getCities(): Promise<string[]> {
    const { data } = await api.get<string[]>('/recommend/cities')
    return data
  },

  async getModels(): Promise<ModelInfo[]> {
    const { data } = await api.get<ModelInfo[]>('/recommend/models')
    return data
  },
}
