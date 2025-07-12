import { useState } from 'react';
import axios from 'axios';
import { createApiUrl } from '@/utils/api';
import { useDispatch, useSelector } from 'react-redux';
import { AppDispatch, RootState } from '@/store/store';
import {
  setConfigLoading,
  setConfigSuccess,
  setConfigError,
  updateLLM,
  updateEmbedder,
  updateMem0Config,
  updateOpenMemory,
  LLMProvider,
  EmbedderProvider,
  Mem0Config,
  OpenMemoryConfig
} from '@/store/configSlice';

// 新增：Mem0服务状态接口
interface Mem0ServiceStatus {
  status: 'healthy' | 'unhealthy' | 'unknown';
  version: string;
  features: string[];
  last_check: string;
  error?: string;
}

// 新增：功能特性配置接口
interface FeatureConfig {
  enabled: boolean;
  config?: Record<string, any>;
}

interface Mem0ServiceConfig {
  base_url: string;
  api_key?: string;
  features: Record<string, FeatureConfig>;
  timeout: number;
}

interface UseConfigApiReturn {
  fetchConfig: () => Promise<void>;
  saveConfig: (config: { openmemory?: OpenMemoryConfig; mem0: Mem0Config }) => Promise<void>;
  saveLLMConfig: (llmConfig: LLMProvider) => Promise<void>;
  saveEmbedderConfig: (embedderConfig: EmbedderProvider) => Promise<void>;
  resetConfig: () => Promise<void>;
  // 新增：Mem0服务相关方法
  fetchMem0ServiceStatus: () => Promise<Mem0ServiceStatus>;
  saveMem0ServiceConfig: (serviceConfig: Mem0ServiceConfig) => Promise<void>;
  testMem0Connection: () => Promise<boolean>;
  getMem0Features: () => Promise<string[]>;
  isLoading: boolean;
  error: string | null;
}

export const useConfig = (): UseConfigApiReturn => {
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const dispatch = useDispatch<AppDispatch>();
  const URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8765";
  const MEM0_URL = process.env.NEXT_PUBLIC_MEM0_API_URL || "http://localhost:8000";
  const buildUrl = (path: string, opts: { trailing?: boolean } = {}) =>
    createApiUrl(URL, path, { trailingSlash: !!opts.trailing });
  
  const fetchConfig = async () => {
    setIsLoading(true);
    dispatch(setConfigLoading());
    
    try {
      const response = await axios.get(buildUrl('/api/v1/config/', { trailing: true }));
      dispatch(setConfigSuccess(response.data));
      setIsLoading(false);
    } catch (err: any) {
      const errorMessage = err.response?.data?.detail || err.message || 'Failed to fetch configuration';
      dispatch(setConfigError(errorMessage));
      setError(errorMessage);
      setIsLoading(false);
      throw new Error(errorMessage);
    }
  };

  const saveConfig = async (config: { openmemory?: OpenMemoryConfig; mem0: Mem0Config }) => {
    setIsLoading(true);
    setError(null);
    
    try {
      const response = await axios.put(buildUrl('/api/v1/config/', { trailing: true }), config);
      dispatch(setConfigSuccess(response.data));
      setIsLoading(false);
      return response.data;
    } catch (err: any) {
      const errorMessage = err.response?.data?.detail || err.message || 'Failed to save configuration';
      dispatch(setConfigError(errorMessage));
      setError(errorMessage);
      setIsLoading(false);
      throw new Error(errorMessage);
    }
  };

  const resetConfig = async () => {
    setIsLoading(true);
    setError(null);
    
    try {
      const response = await axios.post(buildUrl('/api/v1/config/reset/', { trailing: true }));
      dispatch(setConfigSuccess(response.data));
      setIsLoading(false);
      return response.data;
    } catch (err: any) {
      const errorMessage = err.response?.data?.detail || err.message || 'Failed to reset configuration';
      dispatch(setConfigError(errorMessage));
      setError(errorMessage);
      setIsLoading(false);
      throw new Error(errorMessage);
    }
  };

  const saveLLMConfig = async (llmConfig: LLMProvider) => {
    setIsLoading(true);
    setError(null);
    
    try {
      const response = await axios.put(buildUrl('/api/v1/config/mem0/llm/', { trailing: true }), llmConfig);
      dispatch(updateLLM(response.data));
      setIsLoading(false);
      return response.data;
    } catch (err: any) {
      const errorMessage = err.response?.data?.detail || err.message || 'Failed to save LLM configuration';
      setError(errorMessage);
      setIsLoading(false);
      throw new Error(errorMessage);
    }
  };

  const saveEmbedderConfig = async (embedderConfig: EmbedderProvider) => {
    setIsLoading(true);
    setError(null);
    
    try {
      const response = await axios.put(buildUrl('/api/v1/config/mem0/embedder/', { trailing: true }), embedderConfig);
      dispatch(updateEmbedder(response.data));
      setIsLoading(false);
      return response.data;
    } catch (err: any) {
      const errorMessage = err.response?.data?.detail || err.message || 'Failed to save Embedder configuration';
      setError(errorMessage);
      setIsLoading(false);
      throw new Error(errorMessage);
    }
  };

  // 新增：获取Mem0服务状态
  const fetchMem0ServiceStatus = async (): Promise<Mem0ServiceStatus> => {
    setIsLoading(true);
    setError(null);
    
    try {
      const response = await axios.get(`${MEM0_URL}/health`, { timeout: 5000 });
      const healthData = response.data;
      
      // 获取版本信息 - 添加错误处理
      let versionData = { version: 'unknown', features: [] };
      try {
        const versionResponse = await axios.get(`${MEM0_URL}/api/v2/version`, { timeout: 5000 });
        versionData = versionResponse.data;
      } catch (versionError) {
        console.warn('Failed to get version info:', versionError);
      }
      
      // 修复：检查更完整的健康状态
      const isHealthy = healthData.overall_status === 'healthy' || 
                       healthData.status === 'ok';
      
      const status: Mem0ServiceStatus = {
        status: isHealthy ? 'healthy' : 'unhealthy',
        version: versionData.version || 'unknown',
        features: versionData.features || [],
        last_check: new Date().toISOString(),
        error: !isHealthy ? (healthData.error || 'Service not healthy') : undefined
      };
      
      setIsLoading(false);
      return status;
    } catch (err: any) {
      const errorMessage = err.response?.data?.detail || err.message || 'Failed to fetch Mem0 service status';
      setError(errorMessage);
      setIsLoading(false);
      
      return {
        status: 'unhealthy',
        version: 'unknown',
        features: [],
        last_check: new Date().toISOString(),
        error: errorMessage
      };
    }
  };

  // 新增：保存Mem0服务配置
  const saveMem0ServiceConfig = async (serviceConfig: Mem0ServiceConfig) => {
    setIsLoading(true);
    setError(null);
    
    try {
      // 将Mem0服务配置保存到OpenMemory配置中
      const response = await axios.put(`${URL}/api/v1/config/mem0/service/`, serviceConfig);
      setIsLoading(false);
      return response.data;
    } catch (err: any) {
      const errorMessage = err.response?.data?.detail || err.message || 'Failed to save Mem0 service configuration';
      setError(errorMessage);
      setIsLoading(false);
      throw new Error(errorMessage);
    }
  };

  // 新增：测试Mem0连接
  const testMem0Connection = async (): Promise<boolean> => {
    try {
      const response = await axios.get(`${MEM0_URL}/health`, { timeout: 5000 });
      // 修复：检查更完整的健康状态
      return response.status === 200 && (
        response.data.overall_status === 'healthy' || 
        response.data.status === 'ok'
      );
    } catch (err) {
      console.warn('Mem0 connection test failed:', err);
      return false;
    }
  };

  // 新增：获取Mem0功能特性列表
  const getMem0Features = async (): Promise<string[]> => {
    try {
      const response = await axios.get(`${MEM0_URL}/api/v2/version`, { timeout: 5000 });
      return response.data.features || [];
    } catch (err) {
      console.warn('Failed to get Mem0 features:', err);
      // 返回默认的功能特性列表
      return [
        'contextual_add',
        'advanced_search', 
        'criteria_search',
        'custom_categories',
        'memory_export',
        'memory_import',
        'multimodal',
        'graph_memory'
      ];
    }
  };

  return {
    fetchConfig,
    saveConfig,
    saveLLMConfig,
    saveEmbedderConfig,
    resetConfig,
    fetchMem0ServiceStatus,
    saveMem0ServiceConfig,
    testMem0Connection,
    getMem0Features,
    isLoading,
    error
  };
}; 