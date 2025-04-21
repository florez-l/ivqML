// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __ivqML__Model__NeuralNetwork__FeedForward__h__
#define __ivqML__Model__NeuralNetwork__FeedForward__h__

#include <ivqML/Model/Base.h>
#include <ivqML/Model/Functions.h>

#include <vector>

namespace ivqML
{
  namespace Model
  {
    namespace NeuralNetwork
    {
      /**
       */
      template< class _TReal >
      class FeedForward
        : public ivqML::Model::Base< _TReal >
      {
      public:
        using Self       = FeedForward;
        using Superclass = ivqML::Model::Base< _TReal >;
        ivqML_Model_Types;

        using TFunctions = ivqML::Model::Functions< TReal >;
        using TElementwiseActivation = typename TFunctions::TElementwise;
        using TActivation = typename TFunctions::TFunction;
        using TActivationPair = typename TFunctions::TPair;

      public:
        FeedForward( );
        virtual ~FeedForward( ) override;

        virtual void set_size( const TNatural& n ) override;
        virtual void set_input_size( const TNatural& n0 );
        virtual TNatural input_size( ) const override;
        virtual TNatural output_size( ) const override;
        virtual void add_layer( const TNatural& n, const std::string& a );
        TNatural number_of_layers( ) const;

        virtual void init(
          std::function< TReal( ) > g = [](){ return( 0 ); }
          ) override;

        template< class _TX >
        auto operator()( const Eigen::EigenBase< _TX >& X ) const;

        template< class _TX, class _TY >
        TReal gradient(
          TReal* bG,
          const Eigen::EigenBase< _TX >& X,
          const Eigen::EigenBase< _TY >& Y,
          const TReal& l1, const TReal& l2
          ) const;

        virtual void allocate_fitting_buffer( const TNatural& M ) const override;
        virtual void free_fitting_buffer( ) const override;
        /* TODO
           protected:
           struct SBuffer
           {
           TNatural N { 0 };
           TNatural M { 0 };
           TReal*   B { nullptr };

           std::vector< TMatrixMap > Z;
           std::vector< TMatrixMap > A;

           void allocate(
           const std::vector< TNatural >& n,
           const TNatural& m,
           bool keepAZ
           );
           void free( );
           };
        */

      protected:
        void _eval(
          TReal* Ab, TReal* Zb, const TNatural& M, bool offset
          ) const;

        virtual void _to_stream( std::ostream& o ) const override;

      protected:
        std::vector< TNatural >        m_N;
        std::vector< TMatrixMap >      m_W;
        std::vector< TColumnMap >      m_B;
        std::vector< TActivationPair > m_A;

        mutable TReal* m_FittingBuffer { nullptr };

        /* TODO
           mutable SBuffer m_FwdBuf;
           mutable SBuffer m_BwdBuf;
        */
      };
    } // end namespace
  } // end namespace
} // end namespace

#include <ivqML/Model/NeuralNetwork/FeedForward.hxx>

#endif // __ivqML__Model__NeuralNetwork__FeedForward__h__

// eof - $RCSfile$
